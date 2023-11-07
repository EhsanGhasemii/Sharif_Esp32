#include "Mnist_coefficient.hpp"
using namespace dl;

namespace Mnist_coefficient
{
    const static __attribute__((aligned(16))) int16_t statefulpartitionedcall_sequential_conv2d_biasadd_filter_element[] = {
         18213,  31864,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0};

    const static Filter<int16_t> statefulpartitionedcall_sequential_conv2d_biasadd_filter(statefulpartitionedcall_sequential_conv2d_biasadd_filter_element, -15, {1, 1, 1, 2}, {1, 1});
    const Filter<int16_t> *get_statefulpartitionedcall_sequential_conv2d_biasadd_filter()
    {
    	return &statefulpartitionedcall_sequential_conv2d_biasadd_filter;
    }

    const static __attribute__((aligned(16))) int16_t statefulpartitionedcall_sequential_conv2d_biasadd_bias_element[] = {
           173,   -345};

    const static Bias<int16_t> statefulpartitionedcall_sequential_conv2d_biasadd_bias(statefulpartitionedcall_sequential_conv2d_biasadd_bias_element, -7, {2,});
    const Bias<int16_t> *get_statefulpartitionedcall_sequential_conv2d_biasadd_bias()
    {
    	return &statefulpartitionedcall_sequential_conv2d_biasadd_bias;
    }

    const static Activation<int16_t> statefulpartitionedcall_sequential_conv2d_biasadd_activation(ReLU);
    const Activation<int16_t> *get_statefulpartitionedcall_sequential_conv2d_biasadd_activation()
    {
    	return &statefulpartitionedcall_sequential_conv2d_biasadd_activation;
    }

    const static __attribute__((aligned(16))) int16_t fused_gemm_0_filter_element[] = {
         11970, -11098, -18483,   8006};

    const static Filter<int16_t> fused_gemm_0_filter(fused_gemm_0_filter_element, -15, {1, 1, 2, 2}, {1, 1});
    const Filter<int16_t> *get_fused_gemm_0_filter()
    {
    	return &fused_gemm_0_filter;
    }

    const static __attribute__((aligned(16))) int16_t fused_gemm_0_bias_element[] = {
          1722,  -1722};

    const static Bias<int16_t> fused_gemm_0_bias(fused_gemm_0_bias_element, -10, {2,});
    const Bias<int16_t> *get_fused_gemm_0_bias()
    {
    	return &fused_gemm_0_bias;
    }

}
