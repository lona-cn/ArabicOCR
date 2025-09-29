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
~~~bash
git clone https://github.com/yourname/arabic-ocr.git
cd arabic-ocr
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
# ستجد الملفات الثنائية في build/bin/
~~~

### الاستخدام

#### سطر الأوامر
~~~bash
./build/bin/arabic_ocr path/to/input.jpg
~~~

#### مثال واجهة C++
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
