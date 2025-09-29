function IncludeSubDirs(base_dir --[[string]] )
    for _, dir in ipairs(os.dirs(base_dir .. "/*")) do
        local xmake_lua_path = path.join(dir, "xmake.lua")
        if os.exists(xmake_lua_path) then
            includes(xmake_lua_path)
        end
    end
end
