xmake f -c -p windows -a x64 -m "debug" --runtimes="MDd" -vDy &&^
xmake project -k cmake -m "debug" -a "x64" -P . -vDy