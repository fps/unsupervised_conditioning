#include <emscripten/bind.h>
#include <emscripten/val.h>

auto process_net(const emscripten::val &input)
{
  Eigen::Matrix<float, 4, 64> minput;
  Eigen::Matrix<float, 4, 64> moutput;

  anna::op::process(net, minput, 64);  
  return std::array<float, 64>();
}

EMSCRIPTEN_BINDINGS(anna) {
  emscripten::function("process_net", &process_net);
}
