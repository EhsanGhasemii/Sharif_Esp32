#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "dl_tool.hpp"
#include "model_define.hpp"
#include "image.hpp"


extern "C" void app_main(void)
{
Tensor<int16_t> input;
                input.set_element((int16_t *)example_element).set_exponent(input_exponent).set_shape({input_height,input_width,input_channel}).set_auto_free(false);

                clock_t t0 = clock(); 
				MNIST model;
				dl::tool::Latency latency;
                latency.start();
                model.forward(input);
                latency.end();
                latency.print("\nSIGN", "forward");

				float *score = model.l9.get_output().get_element_ptr();
				float max_score = score[0];
				int max_index = 0;

                printf("input : %d\n", example_element[0]);
                for(int i=0; i<2; i++){
                    printf("score[%d] : %.7f\n", i, score[i]);
                }
                printf("=====================\n");

                clock_t t1 = clock();

                printf("time : %016lu \n", (t1 - t0)/1000); 
    
}

















