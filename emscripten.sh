em++ -lembind web.cpp -O3  -I ./../anna/include/ `pkg-config eigen3 --cflags --libs` `pkg-config nlohmann_json --libs --cflags`  -o docs/web.js -s STACK_SIZE=1048576

# em++ -lembind web.cpp -O3  -I ./../anna/include/ `pkg-config eigen3 --cflags --libs` `pkg-config nlohmann_json --libs --cflags`  -o docs/web.js -s ASSERTIONS=2 -s STACK_SIZE=1048576
