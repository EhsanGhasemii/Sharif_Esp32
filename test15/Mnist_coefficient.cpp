#include "Mnist_coefficient.hpp"
using namespace dl;

namespace Mnist_coefficient
{
    const static __attribute__((aligned(16))) int16_t fused_gemm_0_filter_element[] = {
         22554};

    const static Filter<int16_t> fused_gemm_0_filter(fused_gemm_0_filter_element, -19, {1, 1, 1, 1}, {1, 1});
    const Filter<int16_t> *get_fused_gemm_0_filter()
    {
    	return &fused_gemm_0_filter;
    }

    const static __attribute__((aligned(16))) int16_t fused_gemm_0_bias_element[] = {
          2163};

    const static Bias<int16_t> fused_gemm_0_bias(fused_gemm_0_bias_element, -11, {1,});
    const Bias<int16_t> *get_fused_gemm_0_bias()
    {
    	return &fused_gemm_0_bias;
    }

}