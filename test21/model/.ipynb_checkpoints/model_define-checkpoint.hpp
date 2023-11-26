#pragma once
#include <stdint.h>
#include "dl_layer_model.hpp"
#include "dl_layer_base.hpp"
#include "dl_layer_max_pool2d.hpp"
#include "dl_layer_conv2d.hpp"
#include "dl_layer_reshape.hpp"
#include "dl_layer_softmax.hpp"
#include "Mnist_coefficient.hpp"
#include "dl_layer_sigmoid.hpp"

using namespace dl;
using namespace layer;
using namespace Mnist_coefficient;


class MNIST : public Model<int16_t> // Derive the Model class in "dl_layer_model.hpp"
{
private:
	Reshape<int16_t> l1;            // 1. ?
    Conv2D<int16_t> l2; 
//     Reshape<int16_t> l3;
    // MaxPool2D<int16_t> l3; 
    Conv2D<int16_t> l3;
    // MaxPool2D<int16_t> l5; 

    Conv2D<int16_t> l4;
    Conv2D<int16_t> l5;
    Conv2D<int16_t> l6;
    Conv2D<int16_t> l7;
    Conv2D<int16_t> l8;
    Conv2D<int16_t> l9;
    Conv2D<int16_t> l10;
    Conv2D<int16_t> l11;
    Conv2D<int16_t> l12;

    Reshape<int16_t> l13; 
	Conv2D<int16_t> l14;
public:
	Softmax<int16_t> l15; // output layer
	
	/**
	 *  @brief Initialization layers in constructor function
	 *
	 */
    
 
	MNIST () :
		l1(Reshape<int16_t>({28, 28, 1}, "l1_reshape")),
        l2(Conv2D<int16_t>(-7,
                           get_statefulpartitionedcall_sequential_conv2d_biasadd_filter(), 
                           get_statefulpartitionedcall_sequential_conv2d_biasadd_bias(), 
                           get_statefulpartitionedcall_sequential_conv2d_biasadd_activation(),
                           PADDING_VALID,
                           {},
                           1,
                           1,
                           "l2")
          ),
//         l3(Reshape<int16_t>({1, 1, 1}, "l3_reshape")),
        // l3(MaxPool2D<int16_t>({2, 2}, PADDING_VALID, {}, 2, 2, "l3")),
        l3(Conv2D<int16_t>(-8,
                           get_statefulpartitionedcall_sequential_conv2d_1_biasadd_filter(), 
                           get_statefulpartitionedcall_sequential_conv2d_1_biasadd_bias(), 
                           get_statefulpartitionedcall_sequential_conv2d_1_biasadd_activation(),
                           PADDING_VALID,
                           {},
                           1,
                           1,
                           "l4")
          ),
        // l5(MaxPool2D<int16_t>({2, 2}, PADDING_VALID, {}, 2, 2, "l5")),


        l4(Conv2D<int16_t>(-8,
                           get_statefulpartitionedcall_sequential_conv2d_2_biasadd_filter(), 
                           get_statefulpartitionedcall_sequential_conv2d_2_biasadd_bias(), 
                           get_statefulpartitionedcall_sequential_conv2d_2_biasadd_activation(),
                           PADDING_VALID,
                           {},
                           1,
                           1,
                           "l4")
          ),
        l5(Conv2D<int16_t>(-8,
                           get_statefulpartitionedcall_sequential_conv2d_3_biasadd_filter(), 
                           get_statefulpartitionedcall_sequential_conv2d_3_biasadd_bias(), 
                           get_statefulpartitionedcall_sequential_conv2d_3_biasadd_activation(),
                           PADDING_VALID,
                           {},
                           1,
                           1,
                           "l4")
          ),
        l6(Conv2D<int16_t>(-9,
                           get_statefulpartitionedcall_sequential_conv2d_4_biasadd_filter(), 
                           get_statefulpartitionedcall_sequential_conv2d_4_biasadd_bias(), 
                           get_statefulpartitionedcall_sequential_conv2d_4_biasadd_activation(),
                           PADDING_VALID,
                           {},
                           1,
                           1,
                           "l4")
          ),
        l7(Conv2D<int16_t>(-9,
                           get_statefulpartitionedcall_sequential_conv2d_5_biasadd_filter(), 
                           get_statefulpartitionedcall_sequential_conv2d_5_biasadd_bias(), 
                           get_statefulpartitionedcall_sequential_conv2d_5_biasadd_activation(),
                           PADDING_VALID,
                           {},
                           1,
                           1,
                           "l4")
          ),
        l8(Conv2D<int16_t>(-10,
                           get_statefulpartitionedcall_sequential_conv2d_6_biasadd_filter(), 
                           get_statefulpartitionedcall_sequential_conv2d_6_biasadd_bias(), 
                           get_statefulpartitionedcall_sequential_conv2d_6_biasadd_activation(),
                           PADDING_VALID,
                           {},
                           1,
                           1,
                           "l4")
          ),
        l9(Conv2D<int16_t>(-10,
                           get_statefulpartitionedcall_sequential_conv2d_7_biasadd_filter(), 
                           get_statefulpartitionedcall_sequential_conv2d_7_biasadd_bias(), 
                           get_statefulpartitionedcall_sequential_conv2d_7_biasadd_activation(),
                           PADDING_VALID,
                           {},
                           1,
                           1,
                           "l4")
          ),
        l10(Conv2D<int16_t>(-10,
                           get_statefulpartitionedcall_sequential_conv2d_8_biasadd_filter(), 
                           get_statefulpartitionedcall_sequential_conv2d_8_biasadd_bias(), 
                           get_statefulpartitionedcall_sequential_conv2d_8_biasadd_activation(),
                           PADDING_VALID,
                           {},
                           1,
                           1,
                           "l4")
          ),
        l11(Conv2D<int16_t>(-11,
                           get_statefulpartitionedcall_sequential_conv2d_9_biasadd_filter(), 
                           get_statefulpartitionedcall_sequential_conv2d_9_biasadd_bias(), 
                           get_statefulpartitionedcall_sequential_conv2d_9_biasadd_activation(),
                           PADDING_VALID,
                           {},
                           1,
                           1,
                           "l4")
          ),
        l12(Conv2D<int16_t>(-12,
                           get_statefulpartitionedcall_sequential_conv2d_10_biasadd_filter(), 
                           get_statefulpartitionedcall_sequential_conv2d_10_biasadd_bias(), 
                           get_statefulpartitionedcall_sequential_conv2d_10_biasadd_activation(),
                           PADDING_VALID,
                           {},
                           1,
                           1,
                           "l4")
          ),






        l13(Reshape<int16_t>({1, 1, 576}, "l6_reshape")),
		l14(Conv2D<int16_t>(-10,
                           get_fused_gemm_0_filter(),
                           get_fused_gemm_0_bias(),
                           NULL,
                           PADDING_VALID,
                           {},
                           1,
                           1,
                           "l7")
          ),
		l15(Softmax<int16_t>(-14, "l8")){}
    
	/**
	 *  @brief call each layers' build(...) function in sequence
	 *
	 *  @param input
	 */
    
    
    
	void build(Tensor<int16_t> &input)
    {
        this->l1.build(input);
        this->l2.build(this->l1.get_output());
        this->l3.build(this->l2.get_output());  
        this->l4.build(this->l3.get_output());
        this->l5.build(this->l4.get_output());
        this->l6.build(this->l5.get_output());
        this->l7.build(this->l6.get_output());
        this->l8.build(this->l7.get_output()); 

        this->l9.build(this->l8.get_output());
        this->l10.build(this->l9.get_output());
        this->l11.build(this->l10.get_output());
        this->l12.build(this->l11.get_output()); 
        this->l13.build(this->l12.get_output());
        this->l14.build(this->l13.get_output());
        this->l15.build(this->l14.get_output()); 
    }   
    
	/**
	 * @brief call each layers call(...) function in sequence
	 *
	 * @param input 
	 */
    
    
	void call(Tensor<int16_t> &input)
    {
        this->l1.call(input);
        input.free_element();

        this->l2.call(this->l1.get_output());
        this->l1.get_output().free_element();

        this->l3.call(this->l2.get_output());
        this->l2.get_output().free_element();
        
        this->l4.call(this->l3.get_output());
        this->l3.get_output().free_element();

        this->l5.call(this->l4.get_output());
        this->l4.get_output().free_element();
        
        this->l6.call(this->l5.get_output());
        this->l5.get_output().free_element();

        this->l7.call(this->l6.get_output());
        this->l6.get_output().free_element();
        
        this->l8.call(this->l7.get_output());
        this->l7.get_output().free_element();

        this->l9.call(this->l8.get_output());
        this->l8.get_output().free_element();
        
        this->l10.call(this->l9.get_output());
        this->l9.get_output().free_element();

        this->l11.call(this->l10.get_output());
        this->l10.get_output().free_element();
        
        this->l12.call(this->l11.get_output());
        this->l11.get_output().free_element();

        this->l13.call(this->l12.get_output());
        this->l12.get_output().free_element();

        this->l14.call(this->l13.get_output());
        this->l13.get_output().free_element();
        
        this->l15.call(this->l14.get_output());
        this->l14.get_output().free_element();
    }
};
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    