#include <filesystem>
#include <iostream>
#include <mio/mmap.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ArabicOCR.h"


int main(int argc, char* argv[])
{
    SetConsoleOutputCP(CP_UTF8); // 切换 stdout 编码到 UTF-8
    std::cout << std::format("cwd:{}", std::filesystem::current_path().string()) << std::endl;
    auto infer = arabic_ocr::InferContext::Create().value();
    auto ocr = arabic_ocr::OCR::Create(*infer, "assets/det/inference.onnx", "assets/rec/inference.onnx").value();
    cv::Mat mat = cv::imread("assets/imgs/arabic-1.png");
    auto results = ocr->BatchOCR({std::move(mat)});
    for (const auto& result : results)
    {
        std::cout << std::format("{} {}", result.confidence, result.text) << std::endl;
    }
    return 0;
}
