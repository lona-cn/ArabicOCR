#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <expected>
#include <ranges>


#define ARABIC_OCR_DISALLOW_COPY(CLASS) \
    CLASS(const CLASS&) = delete; \
    CLASS& operator=(const CLASS&) = delete;

#define ARABIC_OCR_DISALLOW_MOVE(CLASS) \
    CLASS(CLASS&&) = delete; \
    CLASS& operator=(CLASS&&) = delete;

#define ARABIC_OCR_DISALLOW_COPY_MOVE(CLASS) \
    ARABIC_OCR_DISALLOW_COPY(CLASS); \
    ARABIC_OCR_DISALLOW_MOVE(CLASS);


namespace cv
{
    class Mat;
}

namespace arabic_ocr
{
    enum class ErrorCode:uint8_t
    {
        kOK = 0,
        kUnimplementedError,
        kCustomError,
        kUnknownError,
        kRuntimeError,
        kRangeError,
        kParameterError,
        kIOError,
        kDeviceError,
        kModelError,
    };

    enum class Backend:uint8_t
    {
        kCustom = 0,
        kORT,
    };

    enum class EP:uint8_t
    {
        kAuto = 0,
        kCPU,
        kDML,
        kCUDA,
        kTRT,
        kCoreML,
        kRKNPU,
        kVK,
        kOGL,
        kOCL
    };

    struct Error
    {
        ErrorCode code;
        std::string msg{};
    };

    template <typename T>
    using Result = std::expected<T, Error>;

    struct TextBox
    {
        float x, y, width, height;
        float confidence;
        std::string text;
    };

    template <typename T, typename String = const char*>
        requires(std::convertible_to<String, std::string>)
    Result<T> MakeError(ErrorCode error_code, String&& msg = "")
    {
        return std::unexpected(Error{error_code, std::forward<String>(msg)});
    }


    class InferContext
    {
    public:
        static std::expected<std::unique_ptr<InferContext>, ErrorCode> Create(
            Backend backend = Backend::kORT, EP ep = EP::kCPU) noexcept;

        static std::expected<std::unique_ptr<InferContext>, ErrorCode> Create(
            Backend backend, const std::vector<EP>& eps) noexcept
        {
            for (const auto& ep : eps)
            {
                if (auto result = Create(backend, ep))
                {
                    return std::move(*result);
                }
            }
            return std::unexpected(ErrorCode::kParameterError);
        }

        InferContext() = default;
        ARABIC_OCR_DISALLOW_COPY_MOVE(InferContext)
        virtual ~InferContext() noexcept;
    };

    class OCR
    {
    public:
        static Result<std::unique_ptr<OCR>> Create(
            InferContext& infer_ctx,
            std::span<const uint8_t> det_model_data,
            std::span<const uint8_t> rec_model_data) noexcept;

        static Result<std::unique_ptr<OCR>> Create(
            InferContext& infer_ctx,
            const std::string& det_model_path,
            const std::string& rec_model_path) noexcept;
        OCR() noexcept = default;
        ARABIC_OCR_DISALLOW_COPY_MOVE(OCR)
        virtual ~OCR() noexcept;


        virtual std::vector<TextBox> BatchOCR(const std::vector<cv::Mat>& images) noexcept = 0;
    };
}
