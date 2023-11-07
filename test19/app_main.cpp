#include <stdio.h>
#include <stdlib.h>
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "dl_tool.hpp"
#include "model_define.hpp"


int input_height = 15; 
int input_width = 15;
int input_channel = 1; 
int input_exponent = 0; 


__attribute__((aligned(16))) int16_t example_element[] = { 238, 228, 233, 241, 243, 230, 225, 242, 235, 242, 240, 238, 245, 233, 239, 244, 229, 242, 249, 238, 230, 229, 235, 228, 247, 249, 241, 232, 246, 227, 226, 237, 238, 226, 238, 238, 234, 247, 242, 236, 245, 226, 232, 243, 245, 238, 236, 235, 240, 226, 247, 244, 229, 234, 243, 237, 230, 243, 241, 245, 225, 238, 248, 245, 233, 229, 249, 239, 229, 225, 234, 229, 225, 229, 239, 246, 225, 245, 244, 239, 242, 228, 234, 234, 225, 236, 236, 249, 226, 244, 228, 247, 235, 242, 241, 246, 247, 229, 228, 247, 239, 242, 234, 233, 244, 240, 246, 247, 234, 230, 247, 228, 231, 236, 234, 227, 229, 239, 230, 242, 237, 238, 225, 241, 239, 225, 247, 241, 231, 231, 230, 246, 242, 228, 231, 246, 245, 248, 227, 236, 235, 229, 243, 240, 232, 236, 241, 246, 225, 236, 240, 233, 249, 243, 244, 243, 232, 229, 235, 234, 226, 225, 225, 249, 225, 225, 243, 235, 235, 230, 228, 248, 229, 236, 237, 237, 244, 234, 249, 245, 242, 249, 226, 228, 235, 229, 237, 246, 228, 243, 227, 235, 227, 230, 238, 247, 232, 233, 235, 245, 243, 233, 232, 231, 229, 242, 248, 228, 248, 246, 243, 236, 241, 237, 226, 244, 242, 240, 243, 240, 238, 232, 229, 242, 239
    
    
    //add your input/test image pixels 
};



extern "C" void app_main(void)
{
Tensor<int16_t> input;
                input.set_element((int16_t *)example_element).set_exponent(input_exponent).set_shape({input_height,input_width,input_channel}).set_auto_free(false);

				MNIST model;
				dl::tool::Latency latency;
                latency.start();
                model.forward(input);
                latency.end();
                latency.print("\nSIGN", "forward");

				float *score = model.l8.get_output().get_element_ptr();
				float max_score = score[0];
				int max_index = 0;

                printf("input : %d\n", example_element[0]);
                for(int i=0; i<10; i++){
                    printf("score[%d] : %f\n", i, score[i]);
                }
                printf("=====================\n");
    
                
				/*for(size_t i=0; i<10; i++)
				{
					printf("%f, ", score[i]*100);
					if(score[i] > max_score)
					{
						max_score = score[i];
						max_index = i; 
					}
				}
				printf("\n");

				switch (max_index)
				{
					case 0:
						printf("--0--");
						break;

					case 1:
						printf("--1--");
						break;

					case 2:
						printf("--2--");
						break;

					case 3:
						printf("--3--");
						break;

					case 4:
						printf("--4--");
						break;

					case 5:
						printf("--5--");
						break;

					case 6:
						printf("--6--");
						break;

					case 7:
						printf("--7--");
						break;

					case 8:
						printf("--8--");
						break;

					case 9:
						printf("--9--");
						break;

					default: 
						printf("No result");
				}
				printf("\n");
                */
}

















