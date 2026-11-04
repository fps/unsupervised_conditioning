#include <anna/op/op.hpp>
#include <anna/op/conv1d.hpp>

#define INPUT_CHANNELS 4
#define LAYER_CHANNELS 16
#define OUTPUT_CHANNELS 1
#define MAX_PROCESS_SIZE 128

template<class NextOp>
using Expand = anna::op::linear2<float, LAYER_CHANNELS, INPUT_CHANNELS, MAX_PROCESS_SIZE, NextOp>;

template<int Dilation, class NextOp>
using Layer = anna::op::conv1d<float, LAYER_CHANNELS, LAYER_CHANNELS, 3, Dilation, MAX_PROCESS_SIZE, anna::op::tanh<NextOp>>;

template<class NextOp>
using Block = Layer<1, Layer<2, Layer<4, Layer<8, Layer<LAYER_CHANNELS, Layer<32, Layer<64, Layer< MAX_PROCESS_SIZE, NextOp>>>>>>>>;

template<class Op, class NextOp>
using SkipConnection = anna::op::additive_shortcut<float, Op, NextOp>;

typedef anna::op::output<float, LAYER_CHANNELS, MAX_PROCESS_SIZE> IntermediateOutput;

template<class NextOp>
using Collapse = anna::op::linear2<float, 1, LAYER_CHANNELS, MAX_PROCESS_SIZE, NextOp>;

typedef anna::op::output<float, OUTPUT_CHANNELS, MAX_PROCESS_SIZE> Output;

typedef Expand<SkipConnection<Block<IntermediateOutput>, Block<Collapse<Output>>>> Net;
