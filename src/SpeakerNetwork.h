#ifndef SPEAKER_NETWORK_H
#define SPEAKER_NETWORK_H

// TensorFlow Lite PRIMERO - antes que cualquier cosa que incluya Arduino.h
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

// NO incluir Arduino.h aquí - causa el conflicto con DEFAULT

class SpeakerNetwork {
public:
    SpeakerNetwork();
    ~SpeakerNetwork();
    
    bool begin();
    float* getInputBuffer();           // Buffer float de 80 elementos
    bool predict();
    float* getOutputBuffer();          // Probabilidades en float
    int getNumClasses();               // Numero de clases de salida
    void printMemoryInfo();
    
private:
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input;
    TfLiteTensor* output;
    
    uint8_t* tensor_arena;
    size_t arena_size;
    uint8_t* aligned_arena;
    
    tflite::AllOpsResolver resolver_instance;
    tflite::MicroErrorReporter error_reporter_instance;
    
    // Variables para quantization
    int num_classes;
    bool isQuantized;
    float input_scale;
    int input_zero_point;
    float output_scale;
    int output_zero_point;
    
    // Buffer para output dequantizado
    float output_buffer[10];  // Hasta 10 hablantes
};

#endif