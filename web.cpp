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
Net *net;

void prepare_net(int seed)
{
  /*
  net = new TestNet;
  net->set<0>(Eigen::Matrix<float, 16, 4>::Random());
  net->set<2>(Eigen::Matrix<float, 1, 16>::Random());
  */

  srand(seed);
  
  net = new Net;
  
  net->set<0>(Eigen::Matrix<float, 16, 4>::Random());
  net->set<1>(Eigen::Vector<float, 16>::Random());
  
  std::array<Eigen::Matrix<float, 16, 16>, 3> conv_weights;

  Eigen::Vector<float, LAYER_CHANNELS> biases;

  // Block 1
  
  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().m_op.set<0>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().m_op.set<1>(biases);
  
  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().m_op.set<3>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().m_op.set<4>(biases);

  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().m_op.set<6>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().m_op.set<7>(biases);
  
  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().m_op.set<9>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().m_op.set<10>(biases);

  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().m_op.set<12>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().m_op.set<13>(biases);
  
  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().m_op.set<15>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().m_op.set<16>(biases);

  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().m_op.set<18>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().m_op.set<19>(biases);
  
  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().m_op.set<21>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().m_op.set<22>(biases);

  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().m_op.set<24>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().m_op.set<25>(biases);
  
  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().m_op.set<27>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().m_op.set<28>(biases);

  // Block 2

  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().next().set<0>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().next().set<1>(biases);
  
  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().next().set<3>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().next().set<4>(biases);

  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().next().set<6>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().next().set<7>(biases);
  
  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().next().set<9>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().next().set<10>(biases);

  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().next().set<12>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().next().set<13>(biases);
  
  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().next().set<15>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().next().set<16>(biases);

  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().next().set<18>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().next().set<19>(biases);
  
  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().next().set<21>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().next().set<22>(biases);

  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().next().set<24>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().next().set<25>(biases);
  
  conv_weights = { 0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random(),  0.5 * Eigen::Matrix<float, 16, 16>::Random() };
  net->next().next().next().set<27>(conv_weights);

  biases = 0.5 * Eigen::Vector<float, LAYER_CHANNELS>::Random();
  net->next().next().next().set<28>(biases);

  net->next().next().next().set<30>(0.5 * Eigen::Matrix<float, 1, 16>::Random());
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
    minput(3, col) = mid;
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
