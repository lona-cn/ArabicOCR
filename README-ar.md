# Arabic OCR (C++23 + ONNXRuntime + OpenCV)
[English](./README.md) | [简体中文](./README-zh.md)

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://en.cppreference.com/w/cpp/23"><img alt="C++23" src="https://img.shields.io/badge/C++-23-blue.svg"></a>
  <a href="https://onnxruntime.ai"><img alt="ONNXRuntime" src="https://img.shields.io/badge/ONNXRuntime-Latest-brightgreen.svg"></a>
  <a href="https://opencv.org"><img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.x-orange.svg"></a>
</p>

---

## 🌍 العربية

### المقدمة
مشروع **Arabic OCR** هو مشروع مفتوح المصدر للتعرّف الضوئي على حروف اللغة العربية (OCR)، مُنفّذ بلغة **C++23** ويستخدم **ONNXRuntime** لتنفيذ النماذج و**OpenCV** لمعالجة الصور.
الهدف هو تقديم محرك OCR **سريع، خفيف الوزن، وقابل للنشر** مع واجهة برمجية بسيطة للتكامل.

الرخصة: **Apache-2.0**، ونرحب بمساهمات المجتمع!

### الميزات (قيد التخطيط / التطوير)
- [ ] الاستدلال عبر **ONNXRuntime**
- [ ] معالجة الصور باستخدام **OpenCV** (تحويل إلى تدرج الرمادي، ثنائيات، إزالة التشويش، تصحيح ميلان، تطبيع)
- [ ] دعم التعرّف على النص العربي (المرحلة الأولى: النص المطبوعة؛ لاحقًا: اليدوية وتخطيط الصفحات)
- [ ] واجهة **C++** واضحة وسهلة الاستخدام
- [ ] أداة سطر أوامر لتحويل الصورة → نص
- [ ] اختبارات وحدات وCI (GitHub Actions)

### البناء والتثبيت

#### المتطلبات
- **مترجم ++C23** (GCC 13+ / Clang 16+ / MSVC 2022+)
- [ONNXRuntime](https://onnxruntime.ai/)
- [OpenCV 4.x](https://opencv.org/)
- [CMake 3.20+](https://cmake.org/)
- (اختياري) `pkg-config`, `ninja`

#### خطوات البناء (مثال)

windows_x64_msvc:
```powershell
git clone https://github.com/lona-cn/ArabicOCR.git
cd ArabicOCR/ArabicOCR-Infer
.\scripts\build-windows_x64_msvc.bat
```

### الاستخدام

#### سطر الأوامر
~~~bash
TODO
~~~

#### مثال واجهة C++
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

### النماذج
المشروع يتوقع نموذجًا بصيغة ONNX. حالياً، على المستخدمين إضافة أو تحويل النماذج بأنفسهم. قد يتم توفير روابط لنماذج مدربة مسبقًا في الإصدارات المستقبلية.

مصادر مفيدة:
- ONNX Model Zoo: https://github.com/onnx/models
- Hugging Face: https://huggingface.co/

### المساهمة
- افتح Issues للمشكلات أو الاقتراحات.
- قدّم Pull Requests صغيرة ومركّزة ومرفقة باختبارات إن أمكن.
- الالتزام بنمط كتابة الكود.
- بالمساهمة توافق على ترخيص Apache-2.0 لإسهاماتك.

### الترخيص
هذا المشروع مرخّص بموجب [Apache-2.0](LICENSE).
