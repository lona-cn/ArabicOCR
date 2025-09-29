local target_name = "common"
local kind = "static"
local group_name = "runtime"
local pkgs = {}
local deps = {}
local syslinks = {}
local function callback()
    set_configdir("$(builddir)/config")
    add_includedirs("$(builddir)/config", { public = true })
    add_headerfiles("$(builddir)/config/config.h",
        "$(builddir)/config/export.h")
    add_configfiles("config.h.in", "export.h.in")
    add_extrafiles("config.h.in", "export.h.in")

    on_config(function(target)
        import("core.project.project")
        local project_name = project.name()
        local project_export_name = string.upper(project_name:gsub("-", "_"))
        target:set("configvar", "PROJECT_NAME", project_name)
        local export_in_file_path = path.join(os.scriptdir(), "export.h.in")
        local export_in_file_data = io.readfile(export_in_file_path)
        local find_result = export_in_file_data:find("define_export " .. project_export_name .. "}")
        if find_result == nil then
            io.gsub(export_in_file_path,
                "%${define_export%s+([%w_]+)}",
                "${define_export " .. project_export_name .. "}")
        end
    end)
end
CreateTarget(target_name, kind, os.scriptdir(), group_name, pkgs, deps, syslinks, callback)
