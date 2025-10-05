#include "ArabicOCR.h"
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <algorithm>
#include <numeric>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_float16.h>
#include <onnxruntime_session_options_config_keys.h>
#include <onnxruntime_run_options_config_keys.h>

#ifdef ARABIC_OCR_ONNX_ENABLED
#include <onnx/onnx_pb.h>
#endif
#ifdef WIN32

#endif

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <mio/mmap.hpp>
#include <magic_enum/magic_enum.hpp>

#include "VisionHelper.hpp"

namespace
{
    inline std::vector<std::string> REC_DICT{
        "",
        "!", "#", "$", "%", "&", "'", "(", "+", ",", "-", ".", "/",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        ":", "?", "@",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "_",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "É", "é",
        "ء", "آ", "أ", "ؤ", "إ", "ئ", "ا", "ب", "ة", "ت", "ث", "ج", "ح", "خ", "د", "ذ",
        "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن",
        "ه", "و", "ى", "ي",
        "ً", "ٌ", "ٍ", "َ", "ُ", "ِ", "ّ", "ْ", "ٓ", "ٔ", "ٰ", "ٱ",
        "ٹ", "پ", "چ", "ڈ", "ڑ", "ژ", "ک", "ڭ", "گ", "ں", "ھ", "ۀ", "ہ", "ۂ", "ۃ",
        "ۆ", "ۇ", "ۈ", "ۋ", "ی", "ې", "ے", "ۓ", "ە",
        "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩", "UNK"
    };
#ifdef ARABIC_OCR_ONNX_ENABLED
    void CreateResizeModel(size_t num_channel, size_t width, size_t height)
    {
        onnx::ModelProto model;
        model.set_ir_version(onnx::IR_VERSION);
        model.set_producer_name(
            std::format("ArabicOCR-ResizeTo-{}*{}*{}", num_channel, width, height));
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

    template <class T>
    T align_up(T value, T align)
    {
        static_assert(std::is_integral_v<T>, "align_up only works with integral types");
        return (value + align - 1) / align * align;
    }


    template <typename T>
    void HWC2CHW_BGR2RGB(const cv::Mat& from, T* to) noexcept
    {
        cv::Mat img;
        // force 32FC3
        if (from.type() == CV_32FC3)img = from;
        else from.convertTo(img,CV_32FC3);

        std::vector<cv::Mat> channels_{3};
        size_t width = img.cols, height = img.rows;
        cv::split(img, channels_);
        const size_t num_pixels = width * height;
        for (int c = 0; c < 3; ++c)
        {
            constexpr int channel_mapper[3] = {2, 1, 0};
            const auto src = channels_[channel_mapper[c]].data;
            auto dst = to + num_pixels * c;
            std::memcpy(dst, src, num_pixels * sizeof(T));
        }
    }

    // 将 HWC 的 cv::Mat 转换为 CHW，并写入 Ort::Value
    [[nodiscard]] Ort::Value MatToTensorCHW(const cv::Mat& image, const Ort::MemoryInfo& memory_info)
    {
        auto height = align_up(image.rows, 32);
        auto width = align_up(image.cols, 32);
        cv::Mat img = LetterBox(image, {width, height});
        if (!img.isContinuous())img = img.clone();
        int channels = img.channels();
        // 创建存放 CHW 数据的 buffer
        // 创建 Ort::Value (tensor)
        // ONNX Runtime tensor shape: NCHW (batch=1)
        std::array<int64_t, 4> input_shape{1, channels, height, width};
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            allocator, input_shape.data(), input_shape.size()
        );
        auto info = input_tensor.GetTensorTypeAndShapeInfo();
        // HWC -> CHW
        auto chw_data = input_tensor.GetTensorMutableData<float>();
        HWC2CHW_BGR2RGB<float>(img, chw_data);
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
            if (true)
            {
                session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
                session_options.DisableProfiling();
                session_options.DisablePerSessionThreads();
                //session_options.EnableOrtCustomOps();
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
                    session_options.SetExecutionMode(ORT_SEQUENTIAL);
                    // session_options.SetExecutionMode(ORT_PARALLEL);
                    // session_options.EnableCpuMemArena();
                    // session_options.EnableMemPattern();
                }
            }
            return std::make_unique<Ort::Session>(env_, model_data.data(), model_data.size_bytes(), session_options);
        }

    private:
        Ort::Env env_;
        arabic_ocr::EP ep_;
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
        static arabic_ocr::Result<std::unique_ptr<PPOCRv5>> Create(std::unique_ptr<Ort::Session> det_session,
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
            Ort::IoBinding det_io{*det_session};
            Ort::IoBinding rec_io{*rec_session};
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
                Ort::MemoryInfo output_mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                                             OrtMemType::OrtMemTypeCPUOutput);
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

        std::vector<std::vector<arabic_ocr::TextBox>> BatchOCR(const std::vector<cv::Mat>& images) noexcept override
        {
            if (images.empty()) return {};
            Ort::MemoryInfo input_mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                                        OrtMemType::OrtMemTypeCPUInput);
            std::vector<std::vector<arabic_ocr::TextBox>> ocr_result;
            for (auto& image : images)
            {
                auto tensor{MatToTensorCHW(image, input_mem_info)};
                det_io_.BindInput("x", tensor);
                Ort::RunOptions run_options;
                try
                {
                    det_session_->Run(run_options, det_io_);
                }
                catch (std::exception& err)
                {
                    throw;
                }
                std::vector<cv::Rect> boxes;
                {
                    auto outputs = det_io_.GetOutputValues();
                    //TODO: detect outputs shape
                    auto& output = outputs[0];
                    auto type_and_shape_info = output.GetTensorTypeAndShapeInfo();
                    if (output.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                    {
                        throw std::runtime_error(
                            std::format("unsupported data type:{}",
                                        magic_enum::enum_name(output.GetTensorTypeAndShapeInfo().GetElementType())));
                    }
                    // force data as float
                    auto shape = type_and_shape_info.GetShape();
                    cv::Mat output_img{
                        cv::Size{static_cast<int>(shape[3]), static_cast<int>(shape[2])},CV_32FC1,
                        output.GetTensorMutableData<float>()
                    };
                    cv::Mat output_u8_img;
                    output_img.convertTo(output_u8_img, CV_8UC1, 255.0);
                    std::vector<std::vector<cv::Point>> contours;
                    cv::findContours(output_u8_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                    boxes.reserve(contours.size());
                    for (const auto& contour : contours)
                    {
                        cv::Rect box = cv::boundingRect(contour);
                        auto origin_box = vision_simple::VisionHelper::ScaleCoords(output_u8_img.size(),
                            box, image.size(), true);
                        if (box.width > 3 && box.height > 3 && box.area() > 12)
                            boxes.emplace_back(origin_box);
                    }
                }
                std::ranges::sort(boxes, [](const auto& lhs, const auto& rhs)
                {
                    return lhs.area() > rhs.area();
                });
                // rec
                {
                    auto last_boxes = vision_simple::VisionHelper::FilterByIOU(boxes, 0.5);
                    auto imgs = last_boxes | std::views::transform([&image](const cv::Rect& box)
                    {
                        return image(box);
                    }) | std::ranges::to<std::vector>();
                    auto rec_results = BatchRec(imgs);
                    std::vector<arabic_ocr::TextBox> result;
                    for (auto [i,box] : std::views::enumerate(last_boxes))
                    {
                        auto& rec = rec_results[i];
                        result.emplace_back(box.x, box.y, box.width, box.height, rec.confidence, rec.text);
                    }
                    ocr_result.emplace_back(result);
                }
            }
            return ocr_result;
        }

        std::vector<arabic_ocr::RecResult> BatchRec(const std::vector<cv::Mat>& images) noexcept override
        {
            auto height = this->model_shape_info.rec_input_height;
            auto max_height = std::ranges::max_element(
                images, [](const auto& lhs, const auto& rhs)-> bool
                {
                    return lhs.rows < rhs.rows;
                })->rows;
            auto scale = (float)height / (float)max_height;
            auto width = align_up((int)(std::ranges::max_element(
                                      images, [](const auto& lhs, const auto& rhs)-> bool
                                      {
                                          return lhs.cols < rhs.cols;
                                      })->cols * scale), 320);
            {
                // images
                std::vector<int64_t> rec_shape{
                    static_cast<int64_t>(images.size()),
                    this->model_shape_info.rec_input_num_channels, height, width
                };

                Ort::AllocatorWithDefaultOptions allocator;
                Ort::Value rec_input_tensor = Ort::Value::CreateTensor(
                    allocator, rec_shape.data(), rec_shape.size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
                auto dst = rec_input_tensor.GetTensorMutableData<float>();
                auto stride = width * height;
                for (const auto& [index,image] : std::views::enumerate(images))
                {
                    auto img = LetterBox(image, {width, static_cast<int>(height)});
                    HWC2CHW_BGR2RGB(img, dst + stride * index);
                }
                rec_io_.BindInput("x", rec_input_tensor);
                Ort::RunOptions run_options{};
                rec_session_->Run(run_options, rec_io_);
            }
            // post process
            auto rec_outputs = rec_io_.GetOutputValues();
            auto& value = rec_outputs[0];
            auto info = value.GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();
            auto type = magic_enum::enum_name(info.GetElementType());
            auto ptr = value.GetTensorData<float>();
            auto stride = shape[1] * shape[2];
            std::vector<std::vector<std::pair<int, float>>> result;
            for (auto i : std::views::iota(0, shape[0]))
            {
                std::vector<std::pair<int, float>> indices;
                auto img_ptr = ptr + i * stride;
                for (auto k : std::views::iota(0, shape[1]))
                {
                    auto base_ptr = img_ptr + k * shape[2];
                    auto idx = static_cast<int>(std::ranges::max_element(base_ptr, base_ptr + shape[2]) -
                        base_ptr);
                    auto conf = base_ptr[idx];
                    // it = indices.emplace_hint(it, idx, conf);
                    indices.emplace_back(idx, conf);
                }
                result.emplace_back(std::move(indices));
            }
            std::vector<arabic_ocr::RecResult> rec_results;
            rec_results.reserve(result.size());
            for (auto [i,v] : std::views::enumerate(result))
            {
                for (auto it = v.begin(); it != v.end();)
                {
                    if (it->second < 0.3f)
                        it = v.erase(it);
                    else
                        ++it;
                }
                auto conf_view = v | std::views::transform([](auto& p) { return p.second; });
                auto avg_conf = std::accumulate(conf_view.begin(), conf_view.end(), 0.f) / conf_view.size();
                auto view = v | std::views::transform([](auto& p) { return REC_DICT[p.first]; });
                auto str = std::accumulate(view.begin(), view.end(), std::string{});
                rec_results.emplace_back(avg_conf, str);
            }
            return rec_results;
        }

    private:
        std::unique_ptr<Ort::Session> det_session_;
        std::unique_ptr<Ort::Session> rec_session_;
        Ort::IoBinding det_io_;
        Ort::IoBinding rec_io_;
        ModelShapeInfo model_shape_info;
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
    threading_options.SetGlobalSpinControl(0);
    threading_options.SetGlobalDenormalAsZero();
    std::unique_ptr<InferContext> ctx =
        std::make_unique<InferContextOrt>(Ort::Env{
                                              threading_options, LoggingForward, nullptr,
                                              ORT_LOGGING_LEVEL_INFO, "ArabicOCR"
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
    return PPOCRv5::Create(std::move(det_result.value()), std::move(rec_result.value()));
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

arabic_ocr::OCR::~OCR() noexcept
{
}
