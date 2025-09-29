add_requires("log4cplus")
add_requires("onnxruntime", "yalantinglibs", "opencv-mobile", "mio",
             "magic_enum")
--add_requires("onnx")
--add_requireconfs("onnx", { override = true , version =  "1.17.0" })
--add_requireconfs("onnx.protobuf-cpp", { override = true ,version = "3.11.2" })