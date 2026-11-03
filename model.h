#include <anna/op.hpp>

template<class NextOp>
using Input = anna::op::linear2<float, 16, 4, 128, NextOp>;

template<int Dilation, class NextOp>
using Layer = anna::op::conv1d<float, 16, 16, 3, Dilation, 128, anna::op::tanh<NextOp>>;

template<class NextOp>
using Block = Layer<1, Layer<2, Layer<4, Layer<8, Layer<16, Layer<32, Layer<64, Layer< 128, NextOp>>>>>>>>;

template<class Op, class NextOp>
using SkipConnection = anna::op::additive_shortcut<float, Op, NextOp>;

typedef anna::op::output<float, 16, 128> Output;

// Input<Layer<1, Layer<2, Layer<4, Layer<8, Layer<16, Layer<32, Layer<64, Layer<128, Output>>>>>>>>> net;
Input<SkipConnection<Block<Output>, Block<Output>>> net;

