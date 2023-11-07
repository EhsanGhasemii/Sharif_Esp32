#include "Mnist_coefficient.hpp"
using namespace dl;

namespace Mnist_coefficient
{
    const static __attribute__((aligned(16))) int16_t statefulpartitionedcall_sequential_conv2d_biasadd_filter_element[] = {
          3765,   1827,  -6123,  -2082,  11655,   4697,   4645,  -4894, -14765,  12260,  12333,  13718,  -7093, -17912,  14555,  16662};

    const static Filter<int16_t> statefulpartitionedcall_sequential_conv2d_biasadd_filter(statefulpartitionedcall_sequential_conv2d_biasadd_filter_element, -15, {1, 1, 1, 16}, {1, 1});
    const Filter<int16_t> *get_statefulpartitionedcall_sequential_conv2d_biasadd_filter()
    {
    	return &statefulpartitionedcall_sequential_conv2d_biasadd_filter;
    }

    const static __attribute__((aligned(16))) int16_t statefulpartitionedcall_sequential_conv2d_biasadd_bias_element[] = {
            56,     89,      0,      0,    -65,    113,     -6,      0,      0,   -120,     24,    101,      0,      0,    -69,    -59};

    const static Bias<int16_t> statefulpartitionedcall_sequential_conv2d_biasadd_bias(statefulpartitionedcall_sequential_conv2d_biasadd_bias_element, -7, {16,});
    const Bias<int16_t> *get_statefulpartitionedcall_sequential_conv2d_biasadd_bias()
    {
    	return &statefulpartitionedcall_sequential_conv2d_biasadd_bias;
    }

    const static Activation<int16_t> statefulpartitionedcall_sequential_conv2d_biasadd_activation(ReLU);
    const Activation<int16_t> *get_statefulpartitionedcall_sequential_conv2d_biasadd_activation()
    {
    	return &statefulpartitionedcall_sequential_conv2d_biasadd_activation;
    }

    const static __attribute__((aligned(16))) int16_t statefulpartitionedcall_sequential_conv2d_1_biasadd_filter_element[] = {
          8064,  13153, -19569, -20475,   4762,  18886, -18925, -15353,  -1832,  24551,   1734, -14493,  -8879,   2646, -13456,  12674,   4321,  13148, -27073, -19681,  -8076,  12954,  13848, -21840,   7316, -23827,  -3635,  29766,  -5675,    969,  -6092,  -4350,  -1174, -29325,  -4707,  14732,  17101,   5703,   3421, -20657, -22761,  -9529, -26335, -15028, -21701,  19175,   8741,  14153,   1735,   3574,   8998,  14058,  17616, -24542,  20666,   5840,   4913,   3555,  18111, -17520, -27371,  24821, -14037,  -9264, -15864, -17168,  -3800, -14267,  18932,   6933,  -4139, -23715, -25231, -25768, -15510, -16137,  -7060,  25054,  19128,  11216, -20896,  -4897,  -9955, -23760, -22051, -12746,  22993,   7061, -10899, -25363,  11911,  -1306,  25435,  23402, -27387,  12762, -27408,  14403,  14151, -21930,  15737, -20891, -15493,   4338,  10469,  -4575,  22092,  -5834,  -8939, -26990, -11323, -24670, -18158,  -2582, -18860, -10990, -17080, -21750,   7306, -20888,  16981,  -4716,  23306,  19431, -11494,  26772, -20469,   7269,  13754,  11761,   3170, -24268,   5837,  17518,  22729, -26047,  17518,   2975,   -932,  27144,  -7078,  -3383, -19737,  -1174, -13633,  10763,   5217, -11845,  16613,    276,  13273,   8581, -25550,  -2648,  19164, -28519, -18947,  16896,  25832,  20515,  20854,   8857,  -9553,  -8241, -12688, -22336, -18837, -11346, -12042, -24509,  16408,   -312,   7859,   9189, -12621,  -3991,  13157, -13055,  25674,  15029,  16963,   -133,   9878,   9347, -14031,  -1063,  15409, -14638,  -1307,   -462,   4302, -24352, -11561, -20010, -11048,  22858,  17206,  -9045,  -4736,  14484,  -9534, -14217,  16174, -21368,  15949,  -4386,  13440,  15223,   8499,  27720,  10636,  -4816,   2315, -21107,  -2524,   9182,   4922,  23847,  -6324,   5640,   8104, -20798, -12927, -10553,  18788, -31826, -19939,  12282,  20477,  16293,  14365,   5479, -11489,  10881, -26904,   7103,  12540, -11708,  -7201,   2665,  -6693,  10463,  -3153, -19622,  19623,  16429, -13658,  16248,  23997,  -4056,  12102,  12486,  19864,  18866, -14096,  12309};

    const static Filter<int16_t> statefulpartitionedcall_sequential_conv2d_1_biasadd_filter(statefulpartitionedcall_sequential_conv2d_1_biasadd_filter_element, -16, {1, 1, 16, 16}, {1, 1});
    const Filter<int16_t> *get_statefulpartitionedcall_sequential_conv2d_1_biasadd_filter()
    {
    	return &statefulpartitionedcall_sequential_conv2d_1_biasadd_filter;
    }

    const static __attribute__((aligned(16))) int16_t statefulpartitionedcall_sequential_conv2d_1_biasadd_bias_element[] = {
           132,    248,     -8,      0,      0,      0,      0,    -11,    131,   -111,     -8,     -6,    123,     47,   -110,    118};

    const static Bias<int16_t> statefulpartitionedcall_sequential_conv2d_1_biasadd_bias(statefulpartitionedcall_sequential_conv2d_1_biasadd_bias_element, -8, {16,});
    const Bias<int16_t> *get_statefulpartitionedcall_sequential_conv2d_1_biasadd_bias()
    {
    	return &statefulpartitionedcall_sequential_conv2d_1_biasadd_bias;
    }

    const static Activation<int16_t> statefulpartitionedcall_sequential_conv2d_1_biasadd_activation(ReLU);
    const Activation<int16_t> *get_statefulpartitionedcall_sequential_conv2d_1_biasadd_activation()
    {
    	return &statefulpartitionedcall_sequential_conv2d_1_biasadd_activation;
    }

    const static __attribute__((aligned(16))) int16_t fused_gemm_0_filter_element[] = {
           257,  20343,   -168,  -2334,  -2869,   9373,   5706,  -1879,  -3898,  -4190,   2748,  -6748,  -2149,   4285,  -3076,   2766,  -2431, -21177,   4169,  -4571,   5077,   7553,  -2489,   5517,  -7196,   5823,   3765,   -490,  -5345,  -8624,   3511,  -6653};

    const static Filter<int16_t> fused_gemm_0_filter(fused_gemm_0_filter_element, -14, {1, 1, 16, 2}, {1, 1});
    const Filter<int16_t> *get_fused_gemm_0_filter()
    {
    	return &fused_gemm_0_filter;
    }

    const static __attribute__((aligned(16))) int16_t fused_gemm_0_bias_element[] = {
           227,   -227};

    const static Bias<int16_t> fused_gemm_0_bias(fused_gemm_0_bias_element, -9, {2,});
    const Bias<int16_t> *get_fused_gemm_0_bias()
    {
    	return &fused_gemm_0_bias;
    }

}
