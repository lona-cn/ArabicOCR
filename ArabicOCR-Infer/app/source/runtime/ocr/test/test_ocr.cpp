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
    cv::Mat mat2 = cv::imread("assets/imgs/arabic-2.png");
    // auto rec_results = ocr->BatchRec({mat2});
    //
    // for (auto& result : rec_results)
    // {
    //     std::cout << std::format("{} {}", result.confidence, result.text) << std::endl;
    // }

    // return 0;

    auto results = ocr->BatchOCR({mat});
    for (const auto& result : results)
    {
        for (const auto& text_box : result)
        {
            std::cout << std::format("{} {}", text_box.confidence, text_box.text) << std::endl;
        }
    }
    return 0;
}
