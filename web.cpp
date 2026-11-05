#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <iostream>
#include <cstdlib>

#include "model.h"

// typedef anna::op::linear2<float, 16, 4, 1024, anna::op::linear2<float, 1, 16, 1024, anna::op::output<float, 1, 1024>>> TestNet;
// typedef anna::op::linear2<float, 16, 4, 1024, anna::op::leaky_relu<float, anna::op::linear2<float, 1, 16, 1024, anna::op::output<float, 1, 1024>>>> TestNet;

// anna::op::chain<anna::op::linear2<float, 4, 16, 1024, anna::op::leaky_relu<float, anna::op::linear2<float, 1, 16, 1024, anna::op::output<float, 1, 1024>>>>, Eigen::Matrix<float, 16, 4>, anna::op::leaky_relu<float, anna::op::linear2<float, 1, 16, 1024, anna::op::output<float, 1, 1024>>>> *'
// 'anna::op::linear2<float, 4, 16, 1024,anna::op::leaky_relu<float, anna::op::linear2<float, 1, 16, 1024, anna::op::output<float, 1, 1024>>>> *

// Net *net = new Net;

// TestNet net;
Net *net = nullptr;

void prepare_net(int seed)
{
#include "model_weights.cpp"

  /*
  net = new TestNet;
  net->set<0>(Eigen::Matrix<float, 16, 4>::Random());
  net->set<2>(Eigen::Matrix<float, 1, 16>::Random());
  */

  srand(seed);

  if (net != nullptr) {
    delete net;
  }
  
  net = new Net;

  // INPUT EXPANDER
  
  net->set<0>(m0);
  net->set<1>(b0);

  // Block 1
  
  net->next().next().m_op.set<0>(conv_weights1_1);
  net->next().next().m_op.set<1>(b1_1);

  net->next().next().m_op.set<3>(conv_weights1_2);
  net->next().next().m_op.set<4>(b1_2);

  net->next().next().m_op.set<6>(conv_weights1_3);
  net->next().next().m_op.set<7>(b1_3);

  net->next().next().m_op.set<9>(conv_weights1_4);
  net->next().next().m_op.set<10>(b1_4);

  net->next().next().m_op.set<12>(conv_weights1_5);
  net->next().next().m_op.set<13>(b1_5);

  net->next().next().m_op.set<15>(conv_weights1_6);
  net->next().next().m_op.set<16>(b1_6);

  net->next().next().m_op.set<18>(conv_weights1_7);
  net->next().next().m_op.set<19>(b1_7);

  net->next().next().m_op.set<21>(conv_weights1_8);
  net->next().next().m_op.set<22>(b1_8);

  net->next().next().m_op.set<24>(conv_weights1_9);
  net->next().next().m_op.set<25>(b1_9);

  net->next().next().m_op.set<27>(conv_weights1_10);
  net->next().next().m_op.set<28>(b1_10);

  // Block 2

  net->next().next().next().set<0>(conv_weights2_1);
  net->next().next().next().set<1>(b2_1);

  net->next().next().next().set<3>(conv_weights2_2);
  net->next().next().next().set<4>(b2_2);

  net->next().next().next().set<6>(conv_weights2_3);
  net->next().next().next().set<7>(b2_3);

  net->next().next().next().set<9>(conv_weights2_4);
  net->next().next().next().set<10>(b2_4);

  net->next().next().next().set<12>(conv_weights2_5);
  net->next().next().next().set<13>(b2_5);

  net->next().next().next().set<15>(conv_weights2_6);
  net->next().next().next().set<16>(b2_6);

  net->next().next().next().set<18>(conv_weights2_7);
  net->next().next().next().set<19>(b2_7);

  net->next().next().next().set<21>(conv_weights2_8);
  net->next().next().next().set<22>(b2_8);

  net->next().next().next().set<24>(conv_weights2_9);
  net->next().next().next().set<25>(b2_9);

  net->next().next().next().set<27>(conv_weights2_10);
  net->next().next().next().set<28>(b2_10);

  // OUTPUT COLLAPSER
  net->next().next().next().set<30>(m3);
  net->next().next().next().set<31>(b3);
}

emscripten::val process_net(const emscripten::val &input, float bass, float mid, float treble)
{
  auto input_v = emscripten::convertJSArrayToNumberVector<float>(input);
  // std::cout << "input_v " << input_v.size() << " " << input_v[0] << "\n";
  Eigen::Matrix<float, 4, 1024> minput;

  for (int col = 0; col < 1024; ++col) {
    minput(0, col) = input_v[col];
    minput(1, col) = bass;
    minput(2, col) = mid;
    minput(3, col) = treble;
  }
  
  anna::op::process(*net, minput, 1024);

  emscripten::val output_js = emscripten::val::array();

  for (int index = 0; index < 1024; ++index) {
    output_js.call<void>("push", net->end().m_input(0, index));
  }
  
  return output_js;
}

EMSCRIPTEN_BINDINGS(anna) {
  emscripten::function("process_net", &process_net);
  emscripten::function("prepare_net", &prepare_net);
}
