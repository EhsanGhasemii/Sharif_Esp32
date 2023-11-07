#include <stdio.h>
#include <stdlib.h>
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "dl_tool.hpp"
#include "model_define.hpp"


int input_height = 1; 
int input_width = 1;
int input_channel = 1; 
int input_exponent = 0; 


__attribute__((aligned(16))) int16_t example_element[] = { 0


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

				float *score = model.l3.get_output().get_element_ptr();
				float max_score = score[0];
				int max_index = 0;

                printf("input : %d\n", example_element[0]);
                printf("score : %f\n", score[0]); 
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

















