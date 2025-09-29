#include "ArabicOCR.h"
#include <filesystem>
#include <limits>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_float16.h>
#include <onnxruntime_session_options_config_keys.h>
#include <onnxruntime_run_options_config_keys.h>

#ifdef FALCON_OCR_ONNX_ENABLED
#include <onnx/onnx_pb.h>
#endif
#ifdef WIN32

#endif

#include <opencv2/opencv.hpp>
#include <mio/mmap.hpp>

namespace
{
#ifdef FALCON_OCR_ONNX_ENABLED
    void CreateResizeModel(size_t num_channel, size_t width, size_t height)
    {
        onnx::ModelProto model;
        model.set_ir_version(onnx::IR_VERSION);
        model.set_producer_name(
            std::format("FalconOCR-ResizeTo-{}*{}*{}", num_channel, width, height));
        auto graph = model.mutable_graph();
        graph->set_name("resize_graph");
        auto* input = graph->add_input();
        input->set_name("x");
        // TODO
        // input->mutable_type()->mutable_tensor_type()->
    }
#endif
    cv::Mat LetterBox(const cv::Mat& image, const cv::Size new_shape,
                      const cv::Scalar& color = cv::Scalar(0, 0, 0))
    {
        const int original_w = image.cols;
        const int original_h = image.rows;

        const float r = std::min(static_cast<float>(new_shape.width) / original_w,
                                 static_cast<float>(new_shape.height) / original_h);

        const int new_unpad_w = cvRound(original_w * r);
        const int new_unpad_h = cvRound(original_h * r);

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(new_unpad_w, new_unpad_h));

        const int dw = new_shape.width - new_unpad_w;
        const int dh = new_shape.height - new_unpad_h;

        const int top = dh / 2;
        const int bottom = dh - top;
        const int left = dw / 2;
        const int right = dw - left;

        cv::Mat result;
        cv::copyMakeBorder(resized, result, top, bottom, left, right, cv::BORDER_CONSTANT, color);

        return result;
    }


    // 将 HWC 的 cv::Mat 转换为 CHW，并写入 Ort::Value
    [[nodiscard]] Ort::Value MatToTensorCHW(const cv::Mat& image, const Ort::MemoryInfo& memory_info)
    {
        // 确保图像是连续的
        cv::Mat img;
        if (!image.isContinuous())
        {
            img = image.clone();
        }
        else
        {
            img = image;
        }

        int height = img.rows;
        int width = img.cols;
        int channels = img.channels();

        // 创建存放 CHW 数据的 buffer
        std::vector<float> chw_data(channels * height * width);

        // HWC -> CHW
        const unsigned char* hwc_data = img.ptr<unsigned char>(0);
        size_t hwc_index = 0;

        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                for (int c = 0; c < channels; ++c)
                {
                    size_t chw_index = c * height * width + h * width + w;
                    chw_data[chw_index] = static_cast<float>(hwc_data[hwc_index++]) / 255.0f; // 归一化到[0,1]
                }
            }
        }

        // ONNX Runtime tensor shape: NCHW (batch=1)
        std::array<int64_t, 4> input_shape{1, channels, height, width};

        // 创建 Ort::Value (tensor)
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            chw_data.data(),
            chw_data.size(),
            input_shape.data(),
            input_shape.size()
        );

        return input_tensor;
    }

    Ort::Value MatToTensorCHW(const std::vector<cv::Mat>& images,
                              const Ort::MemoryInfo& memory_info,
                              cv::Size size = {-1, -1})
    {
        if (images.empty())
        {
            throw std::invalid_argument("Input images vector is empty");
        }

        // 1. 确定目标尺寸
        int target_h = size.height;
        int target_w = size.width;

        if (target_h == -1 || target_w == -1)
        {
            target_h = images[0].rows;
            target_w = images[0].cols;
            for (const auto& img : images)
            {
                target_h = std::min(target_h, img.rows);
                target_w = std::min(target_w, img.cols);
            }
        }

        int channels = images[0].channels();
        int batch_size = static_cast<int>(images.size());

        // 2. 分配批量 CHW buffer
        std::vector<float> chw_data(batch_size * channels * target_h * target_w);

        // 3. 转换每张图片
        for (int n = 0; n < batch_size; ++n)
        {
            cv::Mat img = images[n];

            // 转换为连续存储
            if (!img.isContinuous())
            {
                img = img.clone();
            }

            // resize
            if (img.rows != target_h || img.cols != target_w)
            {
                cv::resize(img, img, cv::Size(target_w, target_h));
            }

            // HWC -> CHW
            const unsigned char* hwc_data = img.ptr<unsigned char>(0);
            for (int h = 0; h < target_h; ++h)
            {
                for (int w = 0; w < target_w; ++w)
                {
                    for (int c = 0; c < channels; ++c)
                    {
                        size_t chw_index =
                            n * (channels * target_h * target_w) +
                            c * (target_h * target_w) +
                            h * target_w + w;

                        size_t hwc_index = h * (target_w * channels) + w * channels + c;

                        chw_data[chw_index] = static_cast<float>(hwc_data[hwc_index]) / 255.0f; // normalize
                    }
                }
            }
        }

        // 4. 构造 Ort::Value (tensor), shape = [N, C, H, W]
        std::array<int64_t, 4> input_shape{batch_size, channels, target_h, target_w};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            chw_data.data(),
            chw_data.size(),
            input_shape.data(),
            input_shape.size()
        );

        return input_tensor;
    }

    void LoggingForward(
        void* param, OrtLoggingLevel severity, const char* category,
        const char* logid, const char* code_location, const char* message)
    {
    }


    class InferContextOrt : public arabic_ocr::InferContext
    {
    public:
        explicit InferContextOrt(Ort::Env env, arabic_ocr::EP ep) : InferContext{},
                                                                    env_(std::move(env)), ep_(ep)
        {
        }

        ARABIC_OCR_DISALLOW_COPY_MOVE(InferContextOrt);

        ~InferContextOrt() noexcept override = default;

        [[nodiscard]] Ort::Env& env()
        {
            return env_;
        }

        [[nodiscard]] const Ort::Env& env() const
        {
            return env_;
        }

        /**
         *
         * @param model_data onnx model in bytes
         * @param device_id select inference device otherwise std::numeric_limits<std::size_t>::max() auto detect
         * @return
         */
        arabic_ocr::Result<std::unique_ptr<Ort::Session>> CreateSession(std::span<const uint8_t> model_data,
                                                                        size_t device_id)
        {
            Ort::SessionOptions session_options{};
            // common options
            {
                session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
                session_options.DisableProfiling();
                session_options.DisablePerSessionThreads();
                session_options.EnableOrtCustomOps();
                session_options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1");
                session_options.AddConfigEntry(kOrtSessionOptionsConfigAllowInterOpSpinning, "0");
                session_options.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
                session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "0");
                session_options.AddConfigEntry(kOrtSessionOptionsAvx2PrecisionMode, "1");
                session_options.AddConfigEntry(kOrtSessionOptionsEnableGeluApproximation, "1");
                session_options.SetLogSeverityLevel(OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO);
            }
            // ep options
            switch (ep_)
            {
            case arabic_ocr::EP::kDML:
                break;
            case arabic_ocr::EP::kCUDA:
                break;
            case arabic_ocr::EP::kTRT:
                break;
            case arabic_ocr::EP::kCoreML:
                break;
            case arabic_ocr::EP::kRKNPU:
                break;
            case arabic_ocr::EP::kVK:
                break;
            case arabic_ocr::EP::kOGL:
                break;
            case arabic_ocr::EP::kOCL:
                break;
            case arabic_ocr::EP::kAuto:
            case arabic_ocr::EP::kCPU:
            default:
                {
                    session_options.SetExecutionMode(ORT_PARALLEL);
                    session_options.EnableCpuMemArena();
                    session_options.EnableMemPattern();
                }
            }
            return std::make_unique<Ort::Session>(env_, model_data.data(), model_data.size_bytes(), session_options);
        }

    private:
        Ort::Env env_;
        arabic_ocr::EP ep_;
    };

    struct TextBox
    {
        float x, y, width, height;
        float confidence;
        std::string text;
    };

    class PPOCRv5 : public arabic_ocr::OCR
    {
        PPOCRv5(std::unique_ptr<Ort::Session> det_session,
                std::unique_ptr<Ort::Session> rec_session,
                Ort::IoBinding det_io,
                Ort::IoBinding rec_io) : det_session_(std::move(det_session)),
                                         rec_session_(std::move(rec_session)),
                                         det_io_(std::move(det_io)),
                                         rec_io_(std::move(rec_io)),
                                         model_shape_info{}
        {
            // det model shape info
            {
                auto ts_info = det_session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
                model_shape_info.det_input_element_type = ts_info.GetElementType();
                model_shape_info.det_input_num_channels = ts_info.GetShape()[1];
            }
            // rec model shape info
            {
                auto ts_info = rec_session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
                model_shape_info.rec_input_element_type = ts_info.GetElementType();
                model_shape_info.rec_input_num_channels = ts_info.GetShape()[1];
                model_shape_info.rec_input_height = ts_info.GetShape()[2];
                model_shape_info.rec_output_num_classes = rec_session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo()
                                                                      .GetShape()[2];
            }
        }

        struct ModelShapeInfo
        {
            ONNXTensorElementDataType det_input_element_type;
            ONNXTensorElementDataType rec_input_element_type;
            uint8_t det_input_num_channels;
            uint8_t rec_input_num_channels;
            uint32_t rec_input_height;
            uint32_t rec_output_num_classes;
        };

    public:
        arabic_ocr::Result<std::unique_ptr<PPOCRv5>> Create(std::unique_ptr<Ort::Session> det_session,
                                                            std::unique_ptr<Ort::Session> rec_session)
        {
            if (!det_session)
                return std::unexpected(arabic_ocr::Error{
                    arabic_ocr::ErrorCode::kParameterError, "invalid det session"
                });
            if (!rec_session)
                return std::unexpected(arabic_ocr::Error{
                    arabic_ocr::ErrorCode::kParameterError, "invalid rec session"
                });
            Ort::IoBinding det_io{*det_session_};
            Ort::IoBinding rec_io{*rec_session_};
            {
                // Ort::UnownedIoBinding det_io{};
                auto setup_output = [](const Ort::Session& session, const Ort::MemoryInfo& memory_info,
                                       Ort::IoBinding& io)
                {
                    for (const auto& ort_value_info : session.GetOutputs())
                    {
                        const auto& name = ort_value_info.Name();
                        auto onnx_type = ort_value_info.TypeInfo().GetONNXType();
                        auto type_info = ort_value_info.TypeInfo();
                        auto tensor_type_and_shape_info = type_info.GetTensorTypeAndShapeInfo();
                        auto shapes = tensor_type_and_shape_info.GetShape();
                        auto symbols = tensor_type_and_shape_info.GetSymbolicDimensions();
                        auto element_type = tensor_type_and_shape_info.GetElementType();
                        io.BindOutput(name.data(), memory_info);
                    }
                };
                setup_output(*det_session, output_mem_info, det_io);
                setup_output(*rec_session, output_mem_info, rec_io);
            }
            return std::unique_ptr<PPOCRv5>{
                new PPOCRv5{
                    std::move(det_session), std::move(rec_session),
                    std::move(det_io), std::move(rec_io)
                }
            };
        }

        ~PPOCRv5() noexcept override = default;

        std::vector<cv::Mat> BatchDet(const std::vector<cv::Mat>& images) noexcept
        {
            if (images.empty()) return {};
            // auto is_same_size = std::ranges::all_of(images, [w=images[0].cols,h = images[0].rows](const auto& image)
            // {
            //     return image.rows == w && image.cols == h;
            // });
            for (auto& image : images)
            {
                auto tensor{MatToTensorCHW(image, input_mem_info)};
                det_io_.BindInput("x", tensor);
                Ort::RunOptions run_options;
                det_session_->Run(run_options, det_io_);
                rec_session_->Run(run_options, rec_io_);
            }
        }

        std::vector<TextBox> BatchRec(const std::vector<cv::Mat>& images) noexcept
        {
        }

        std::vector<TextBox> BatchOCR(const std::vector<cv::Mat>& images) noexcept
        {
            BatchDet(images);
        }

    private:
        std::unique_ptr<Ort::Session> det_session_;
        std::unique_ptr<Ort::Session> rec_session_;
        Ort::IoBinding det_io_;
        Ort::IoBinding rec_io_;
        ModelShapeInfo model_shape_info;

        static inline const auto input_mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                                             OrtMemType::OrtMemTypeCPUInput);
        static inline const auto output_mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                                              OrtMemType::OrtMemTypeCPUOutput);
    };
}

std::expected<std::unique_ptr<arabic_ocr::InferContext>, arabic_ocr::ErrorCode> arabic_ocr::InferContext::Create(
    Backend backend,
    EP ep) noexcept
{
    if (backend != Backend::kORT) return std::unexpected{ErrorCode::kParameterError};
    static const std::vector<EP> kSupportedEPs{EP::kCPU, EP::kCoreML, EP::kCUDA, EP::kTRT, EP::kDML};
    // filter supported execution providers 
    if (std::ranges::find(kSupportedEPs, ep) == kSupportedEPs.end())
        return std::unexpected(ErrorCode::kParameterError);
    Ort::ThreadingOptions threading_options{};
    auto cpu_cores = std::max(1u, std::thread::hardware_concurrency());
    threading_options.SetGlobalInterOpNumThreads(std::max(1, static_cast<int>(0.75f * cpu_cores)));
    threading_options.SetGlobalIntraOpNumThreads(std::max(1, static_cast<int>(0.25f * cpu_cores)));
    //
    threading_options.SetGlobalSpinControl(0);
    threading_options.SetGlobalDenormalAsZero();
    std::unique_ptr<InferContext> ctx =
        std::make_unique<InferContextOrt>(Ort::Env{
                                              threading_options, LoggingForward, nullptr,
                                              ORT_LOGGING_LEVEL_INFO, "FalconOCR"
                                          }, ep);
    return std::move(ctx);
}

arabic_ocr::InferContext::~InferContext() noexcept = default;

arabic_ocr::Result<std::unique_ptr<arabic_ocr::OCR>> arabic_ocr::OCR::Create(InferContext& infer_ctx,
                                                                             std::span<const uint8_t> det_model_data,
                                                                             std::span<const uint8_t> rec_model_data)
    noexcept
{
    auto& ort_infer_ctx = dynamic_cast<InferContextOrt&>(infer_ctx);
    auto det_result = ort_infer_ctx.CreateSession(det_model_data,
                                                  std::numeric_limits<std::size_t>::max());
    if (!det_result)return std::unexpected(std::move(det_result.error()));
    auto rec_result = ort_infer_ctx.CreateSession(rec_model_data,
                                                  std::numeric_limits<std::size_t>::max());
    if (!rec_result)return std::unexpected(std::move(rec_result.error()));
}

arabic_ocr::Result<std::unique_ptr<arabic_ocr::OCR>> arabic_ocr::OCR::Create(InferContext& infer_ctx,
                                                                             const std::string& det_model_path,
                                                                             const std::string& rec_model_path) noexcept
{
    if (std::filesystem::is_directory(det_model_path) || !std::filesystem::exists(det_model_path))
    {
        return MakeError<std::unique_ptr<OCR>>(ErrorCode::kIOError,
                                               std::format("det_model_path not exist or is a directory"));
    }
    if (std::filesystem::is_directory(rec_model_path) || !std::filesystem::exists(rec_model_path))
    {
        return MakeError<std::unique_ptr<OCR>>(ErrorCode::kIOError,
                                               std::format("rec_model_path not exist or is a directory"));
    }
    std::error_code ec;
    auto det_mmap = mio::make_mmap_source(det_model_path, ec);
    if (ec)
    {
        return MakeError<std::unique_ptr<OCR>>(ErrorCode::kIOError,
                                               std::format("unable to mmap det_model_path:{}", ec.message()));
    }
    auto rec_mmap = mio::make_mmap_source(rec_model_path, ec);
    if (ec)
    {
        return MakeError<std::unique_ptr<OCR>>(ErrorCode::kIOError,
                                               std::format("unable to mmap det_model_path:{}", ec.message()));
    }
    return Create(infer_ctx,
                  std::span(reinterpret_cast<const uint8_t*>(det_mmap.data()), det_mmap.size()),
                  std::span(reinterpret_cast<const uint8_t*>(rec_mmap.data()), rec_mmap.size()));
}
