#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "esp_timer.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "dl_tool.hpp"
#include "model_define.hpp"
#include "image.hpp"


extern "C" void app_main(void) {
    Tensor<int16_t> input;
    
    input.set_element((int16_t *)example_element)
         .set_exponent(input_exponent)
         .set_shape({input_height,input_width,input_channel})
         .set_auto_free(false);
        
    MNIST model;
    dl::tool::Latency latency;
    latency.start();

    // start the timer
    uint64_t start_time = esp_timer_get_time();
    
    model.forward(input);

    // stop the timer
    uint64_t end_time = esp_timer_get_time();
    
    // claculate the elapsed time
    uint64_t elapsed = end_time - start_time; 
    
    // convert time format to float 
    float elapsed_time = float(elapsed);

    latency.end();
    latency.print("\nSIGN", "forward");
    
    float *score = model.l15.get_output().get_element_ptr();
    float max_score = score[0];
    int max_index = 0;
    
    printf("input : %d\n", example_element[0]);
    for(int i=0; i<10; i++){
        printf("score[%d] : %.7f\n", i, score[i]);
    }
    printf("=====================\n");
    
    printf("Elapsed time : %.3f miliseconds\n", elapsed_time/1000); 
    
}

















