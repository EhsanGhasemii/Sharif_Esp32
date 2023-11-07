#pragma once

#include <stdint.h>
#include "dl_constant.hpp"

namespace Mnist_coefficient
{
    const dl::Filter<int16_t> *get_fused_gemm_0_filter();
    const dl::Bias<int16_t> *get_fused_gemm_0_bias();
}
