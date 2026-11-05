#include <anna/op/op.hpp>
#include <anna/op/conv1d.hpp>

#define INPUT_CHANNELS 4
#define LAYER_CHANNELS 16
#define OUTPUT_CHANNELS 1
#define MAX_PROCESS_SIZE 1024

template<class NextOp>
using Expand = anna::op::linear2<float, LAYER_CHANNELS, INPUT_CHANNELS, MAX_PROCESS_SIZE, anna::op::vector_add<float, 16, NextOp>>;

template<int Dilation, class NextOp>
using Conv = anna::op::conv1d<float, LAYER_CHANNELS, LAYER_CHANNELS, 3, Dilation, MAX_PROCESS_SIZE, NextOp>;

template<int Dilation, class NextOp>
using Layer = Conv<Dilation, anna::op::vector_add<float, LAYER_CHANNELS, anna::op::leaky_relu<float, NextOp>>>;

template<class NextOp>
using Block = Layer<1, Layer<2, Layer<4, Layer<8, Layer<16, Layer<32, Layer<64, Layer<128, Layer<512, Layer<1024, NextOp>>>>>>>>>>;

template<class Op, class NextOp>
using SkipConnection = anna::op::additive_shortcut<float, Op, NextOp>;

typedef anna::op::output<float, LAYER_CHANNELS, MAX_PROCESS_SIZE> IntermediateOutput;

template<class NextOp>
using Collapse = anna::op::linear2<float, 1, LAYER_CHANNELS, MAX_PROCESS_SIZE, anna::op::vector_add<float, 1, NextOp>>;

typedef anna::op::output<float, OUTPUT_CHANNELS, MAX_PROCESS_SIZE> Output;

typedef Expand<SkipConnection<Block<IntermediateOutput>, Block<Collapse<Output>>>> Net;
