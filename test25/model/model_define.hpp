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
    // MaxPool2D<int16_t> l3; 
    Conv2D<int16_t> l3;
    MaxPool2D<int16_t> l4; 
    // Reshape<int16_t> l4;

    // Conv2D<int16_t> l4;
    Conv2D<int16_t> l5;
    MaxPool2D<int16_t> l6; 

    Conv2D<int16_t> l7;
    Conv2D<int16_t> l8;
    MaxPool2D<int16_t> l9; 

    Conv2D<int16_t> l10;
    Conv2D<int16_t> l11;
    MaxPool2D<int16_t> l12; 

    Conv2D<int16_t> l16;
    Conv2D<int16_t> l17;
    MaxPool2D<int16_t> l18;

    // Conv2D<int16_t> l19;
    // Conv2D<int16_t> l20;




    Reshape<int16_t> l13; 
	Conv2D<int16_t> l14;
public:
	Softmax<int16_t> l15; // output layer
	
	/**
	 *  @brief Initialization layers in constructor function
	 *
	 */
    
 
	MNIST () :
		l1(Reshape<int16_t>({1, 12000, 1}, "l1_reshape")),
        l2(Conv2D<int16_t>(-11,
                           get_statefulpartitionedcall_model_conv2d_conv2d_filter(), 
                           get_statefulpartitionedcall_model_conv2d_conv2d_bias(), 
                           get_statefulpartitionedcall_model_conv2d_conv2d_activation(),
                           PADDING_VALID,
                           {},
                           1,
                           2,
                           "l2")
          ),
//         l3(Reshape<int16_t>({1, 1, 1}, "l3_reshape")),
        // l3(MaxPool2D<int16_t>({2, 2}, PADDING_VALID, {}, 2, 2, "l3")),
        l3(Conv2D<int16_t>(-11,
                           get_statefulpartitionedcall_model_conv2d_1_conv2d_filter(), 
                           get_statefulpartitionedcall_model_conv2d_1_conv2d_bias(), 
                           get_statefulpartitionedcall_model_conv2d_1_conv2d_activation(),
                           PADDING_VALID,
                           {},
                           1,
                           2,
                           "l4")
          ),
        l4(MaxPool2D<int16_t>({1, 110}, PADDING_VALID, {}, 1, 110, "l4")),



        l5(Conv2D<int16_t>(-11,
                           get_statefulpartitionedcall_model_conv2d_2_conv2d_filter(), 
                           get_statefulpartitionedcall_model_conv2d_2_conv2d_bias(), 
                           get_statefulpartitionedcall_model_conv2d_2_conv2d_activation(),
                           PADDING_SAME_END,
                           {},
                           1,
                           1,
                           "l4")
          ),

        l6(MaxPool2D<int16_t>({1, 2}, PADDING_VALID, {}, 1, 2, "l6")),

        l7(Conv2D<int16_t>(-11,
                           get_statefulpartitionedcall_model_conv2d_3_conv2d_filter(), 
                           get_statefulpartitionedcall_model_conv2d_3_conv2d_bias(), 
                           get_statefulpartitionedcall_model_conv2d_3_conv2d_activation(),
                           PADDING_SAME_END,
                           {},
                           1,
                           1,
                           "l7")
          ),
        l8(Conv2D<int16_t>(-11,
                           get_statefulpartitionedcall_model_conv2d_4_conv2d_filter(), 
                           get_statefulpartitionedcall_model_conv2d_4_conv2d_bias(), 
                           get_statefulpartitionedcall_model_conv2d_4_conv2d_activation(),
                           PADDING_SAME_END,
                           {},
                           1,
                           1,
                           "l8")
          ),
        l9(MaxPool2D<int16_t>({1, 2}, PADDING_VALID, {}, 1, 2, "l9")),

        l10(Conv2D<int16_t>(-12,
                           get_statefulpartitionedcall_model_conv2d_5_conv2d_filter(), 
                           get_statefulpartitionedcall_model_conv2d_5_conv2d_bias(), 
                           get_statefulpartitionedcall_model_conv2d_5_conv2d_activation(),
                           PADDING_SAME_END,
                           {},
                           1,
                           1,
                           "l10")
          ),

        l11(Conv2D<int16_t>(-11,
                           get_statefulpartitionedcall_model_conv2d_6_conv2d_filter(), 
                           get_statefulpartitionedcall_model_conv2d_6_conv2d_bias(), 
                           get_statefulpartitionedcall_model_conv2d_6_conv2d_activation(),
                           PADDING_SAME_END,
                           {},
                           1,
                           1,
                           "l4")
          ),
        l12(MaxPool2D<int16_t>({1, 2}, PADDING_VALID, {}, 1, 2, "l12")),

        l16(Conv2D<int16_t>(-12,
                           get_statefulpartitionedcall_model_conv2d_7_conv2d_filter(), 
                           get_statefulpartitionedcall_model_conv2d_7_conv2d_bias(), 
                           get_statefulpartitionedcall_model_conv2d_7_conv2d_activation(),
                           PADDING_SAME_END,
                           {},
                           1,
                           1,
                           "l16")
          ),

        l17(Conv2D<int16_t>(-11,
                           get_statefulpartitionedcall_model_conv2d_8_conv2d_filter(), 
                           get_statefulpartitionedcall_model_conv2d_8_conv2d_bias(), 
                           get_statefulpartitionedcall_model_conv2d_8_conv2d_activation(),
                           PADDING_SAME_END,
                           {},
                           1,
                           1,
                           "l17")
        ),
        l18(MaxPool2D<int16_t>({1, 2}, PADDING_VALID, {}, 1, 2, "l18")),

        // l19(Conv2D<int16_t>(-12,
        //                    get_statefulpartitionedcall_model_conv2d_9_conv2d_filter(), 
        //                    get_statefulpartitionedcall_model_conv2d_9_conv2d_bias(), 
        //                    get_statefulpartitionedcall_model_conv2d_9_conv2d_activation(),
        //                    PADDING_SAME_END,
        //                    {},
        //                    1,
        //                    1,
        //                    "l19")
        //   ),

        // l20(Conv2D<int16_t>(-11,
        //                    get_statefulpartitionedcall_model_conv2d_10_conv2d_filter(), 
        //                    get_statefulpartitionedcall_model_conv2d_10_conv2d_bias(), 
        //                    get_statefulpartitionedcall_model_conv2d_10_conv2d_activation(),
        //                    PADDING_SAME_END,
        //                    {},
        //                    1,
        //                    1,
        //                    "l20")
        // ),





        l13(Reshape<int16_t>({1, 1, 16}, "l6_reshape")),
		l14(Conv2D<int16_t>(-13,
                           get_fused_gemm_0_filter(),
                           get_fused_gemm_0_bias(),
                           NULL,
                           PADDING_VALID,
                           {},
                           1,
                           1,
                           "l7")
          ),
		l15(Softmax<int16_t>(-15, "l8")){}
    
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

        this->l16.build(this->l12.get_output());
        this->l17.build(this->l16.get_output()); 
        this->l18.build(this->l17.get_output()); 

        // this->l19.build(this->l18.get_output()); 
        // this->l20.build(this->l19.get_output()); 
        
        
        this->l13.build(this->l18.get_output());
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

        
        this->l16.call(this->l12.get_output());
        this->l12.get_output().free_element();
        
        this->l17.call(this->l16.get_output());
        this->l16.get_output().free_element();
        
        this->l18.call(this->l17.get_output());
        this->l17.get_output().free_element();

        // this->l19.call(this->l18.get_output());
        // this->l18.get_output().free_element();
        
        // this->l20.call(this->l19.get_output());
        // this->l19.get_output().free_element();
        

        
        this->l13.call(this->l18.get_output());
        this->l18.get_output().free_element();

        this->l14.call(this->l13.get_output());
        this->l13.get_output().free_element();
        
        this->l15.call(this->l14.get_output());
        this->l14.get_output().free_element();
    }
};
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    