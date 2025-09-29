# Arabic OCR (C++23 + ONNXRuntime + OpenCV)
[English](./README.md) | [Arabic](./README-ar.md)

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://en.cppreference.com/w/cpp/23"><img alt="C++23" src="https://img.shields.io/badge/C++-23-blue.svg"></a>
  <a href="https://onnxruntime.ai"><img alt="ONNXRuntime" src="https://img.shields.io/badge/ONNXRuntime-Latest-brightgreen.svg"></a>
  <a href="https://opencv.org"><img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.x-orange.svg"></a>
</p>

---

## 🌍 中文

### 项目简介
**Arabic OCR** 是一个开源的阿拉伯语光学字符识别（OCR）项目，使用 **C++23** 开发，采用 **ONNXRuntime** 做模型推理，使用 **OpenCV** 做图像预处理。
目标是打造一个 **快速、轻量、易集成** 的阿拉伯语 OCR 引擎，适合嵌入到 C++ 项目或作为命令行工具使用。

本项目遵循 **Apache-2.0** 协议，欢迎社区贡献。

### 功能（规划中）
- [ ] 基于 **ONNXRuntime** 的深度学习推理
- [ ] 基于 **OpenCV** 的图像预处理（灰度化、二值化、去噪、倾斜校正、归一化）
- [ ] 阿拉伯语文本识别（初期以印刷体为主；后续考虑手写与复杂版面）
- [ ] 简洁的 **C++ API**
- [ ] 命令行工具（图像 → 文本）
- [ ] 单元测试与持续集成

### 构建与安装

#### 依赖
- **C++23 编译器**（GCC 13+ / Clang 16+ / MSVC 2022+）
- [ONNXRuntime](https://onnxruntime.ai/)
- [OpenCV 4.x](https://opencv.org/)
- [CMake 3.20+](https://cmake.org/)
- 可选：`pkg-config`, `ninja`

#### 构建示例
~~~bash
git clone https://github.com/yourname/arabic-ocr.git
cd arabic-ocr
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
# 可执行文件位于 build/bin/
~~~

### 使用方法

#### 命令行
~~~bash
./build/bin/arabic_ocr path/to/input.jpg
~~~

#### C++ 示例
~~~cpp
#include "ocr/ocr.h"

int main() {
    OCR::ModelOptions opts;
    opts.model_path = "models/arabic_recog.onnx";
    OCR::Engine engine(opts);

    std::string text = engine.recognize_from_file("test_images/page1.jpg");
    std::cout << text << std::endl;
    return 0;
}
~~~

### 模型
本项目使用 ONNX 格式的识别模型。当前需要用户自行提供或转换模型，未来可能在仓库或 Releases 中添加预训练模型下载链接。

推荐资源：
- ONNX Model Zoo: https://github.com/onnx/models
- Hugging Face: https://huggingface.co/

### 贡献
欢迎提交 Issue 和 Pull Request。请尽量保持 PR 小而专注，并在可能情况下包含测试。贡献自动适用 Apache-2.0 许可条款。

### 许可证
本项目基于 [Apache License 2.0](LICENSE) 许可。
