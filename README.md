# Arabic OCR (C++23 + ONNXRuntime + OpenCV)
[Arabic](./README-ar.md) | [简体中文](./README-zh.md)


<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://en.cppreference.com/w/cpp/23"><img alt="C++23" src="https://img.shields.io/badge/C++-23-blue.svg"></a>
  <a href="https://onnxruntime.ai"><img alt="ONNXRuntime" src="https://img.shields.io/badge/ONNXRuntime-Latest-brightgreen.svg"></a>
  <a href="https://opencv.org"><img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.x-orange.svg"></a>
</p>


---

## 🌍 English

### Introduction
**Arabic OCR** is an open-source project for Optical Character Recognition (OCR) of Arabic text, built with **C++23**, **ONNXRuntime**, and **OpenCV**.
The goal is to provide a **fast, lightweight, and portable OCR engine** for Arabic script.

Licensed under **Apache-2.0**, community contributions are welcome!

### Features (Planned)
- [ ] Deep learning inference with **ONNXRuntime**
- [ ] **OpenCV** preprocessing (grayscale, binarization, skew correction, etc.)
- [ ] **Arabic character recognition** (initial focus: printed text)
- [ ] Clean **C++ API** for easy integration
- [ ] Command-line tool for image → text

### Build & Install
#### Dependencies
- **C++23 compiler** (GCC 13 / Clang 16 / MSVC 2022+)
- [ONNXRuntime](https://onnxruntime.ai/)
- [OpenCV 4.x](https://opencv.org/)
- [CMake 3.20+](https://cmake.org/)

#### Build

windows_x64_msvc:
```powershell
git clone https://github.com/lona-cn/ArabicOCR.git
cd ArabicOCR/ArabicOCR-Infer
.\scripts\build-windows_x64_msvc.bat
```


#### C++ example
~~~cpp
#include <filesystem>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <windows.h>

#include "ArabicOCR.h"


int main(int argc, char* argv[])
{
    SetConsoleOutputCP(CP_UTF8);
    std::cout << std::format("cwd:{}", std::filesystem::current_path().string()) << std::endl;
    auto infer = arabic_ocr::InferContext::Create().value();
    auto ocr = arabic_ocr::OCR::Create(*infer, "assets/det/inference.onnx", "assets/rec/inference.onnx").value();
    cv::Mat mat = cv::imread("assets/imgs/arabic-1.png");

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

~~~
