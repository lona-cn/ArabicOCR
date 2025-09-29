local target_name = "ocr"
local kind = "static"
local group_name = "runtime"
local pkgs = {
    "onnxruntime", "yalantinglibs", "opencv-mobile", "mio", "magic_enum"
}
local deps = {}
local syslinks = {}
CreateTarget(target_name, kind, os.scriptdir(), group_name, pkgs, deps,
             syslinks, callback)
