#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <iostream>

#include "model.h"

Net net;

emscripten::val process_net(const emscripten::val &input)
{
  auto input_v = emscripten::convertJSArrayToNumberVector<float>(input);

  Eigen::Matrix<float, 4, 1024> minput;

  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 1024; ++col) {
      minput(row, col) = input_v[row * 1024 + col];
    }
  }
    
  //std::cout << "hah! " << input_v[0] << "\n";
  
  anna::op::process(net, minput, 1024);  
  // return std::array<float, 1024>();

  emscripten::val output_js = emscripten::val::array();

  for (int index = 0; index < 1024; ++index) {
    output_js.call<void>("push", net.end().m_input(0, index));
  }
  
  return output_js;
}

EMSCRIPTEN_BINDINGS(anna) {
  emscripten::function("process_net", &process_net);
}
