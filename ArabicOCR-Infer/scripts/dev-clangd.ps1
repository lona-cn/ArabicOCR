xmake f -c -p windows -a x64 -m "debug" --toolchain=clang-cl --runtimes="MDd" -vDy
xmake project -k compile_commands
