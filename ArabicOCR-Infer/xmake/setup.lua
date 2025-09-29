local function dir_has_lua_files(dir --[[string]] )
    local path_regex = path.join(dir, "*.lua")
    local files = os.files(path_regex)
    return #files > 0
end

local function include_if_lua_exists(dir --[[string]] )
    local path_regex = path.join(dir, "*.lua")
    if dir_has_lua_files(dir) then
        includes(path_regex)
    else
    end
end

local function include_if_lua_exist(file --[[string]] )
    if os.exists(file) then includes(file) end
end

includes("project.lua")
SetupProject()
-- includes("funcs/*.lua")
-- includes("options.lua")
-- includes("rules/*.lua")
-- includes("repo.lua")
include_if_lua_exists("funcs")
include_if_lua_exist("options.lua")
include_if_lua_exists("rules")
include_if_lua_exist("repo.lua")
local modules_dir = path.join(os.scriptdir(), "modules");
if dir_has_lua_files(modules_dir) then add_moduledirs(modules_dir) end
local plugins_dir = path.join(os.scriptdir(), "plugins");
if dir_has_lua_files(plugins_dir) then add_plugindirs(plugins_dir) end
includes(path.join(os.projectdir(), "app", "includes.lua"))
