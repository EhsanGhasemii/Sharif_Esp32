#pragma once

#include <stdint.h>
#include "dl_constant.hpp"

namespace Mnist_coefficient
{
    const dl::Filter<int16_t> *get_statefulpartitionedcall_sequential_conv2d_biasadd_filter();
    const dl::Bias<int16_t> *get_statefulpartitionedcall_sequential_conv2d_biasadd_bias();
    const dl::Activation<int16_t> *get_statefulpartitionedcall_sequential_conv2d_biasadd_activation();
    const dl::Filter<int16_t> *get_statefulpartitionedcall_sequential_conv2d_1_biasadd_filter();
    const dl::Bias<int16_t> *get_statefulpartitionedcall_sequential_conv2d_1_biasadd_bias();
    const dl::Activation<int16_t> *get_statefulpartitionedcall_sequential_conv2d_1_biasadd_activation();
    const dl::Filter<int16_t> *get_fused_gemm_0_filter();
    const dl::Bias<int16_t> *get_fused_gemm_0_bias();
}