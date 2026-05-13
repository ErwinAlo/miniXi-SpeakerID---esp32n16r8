// ============================================================
// SpeakerNetwork.cpp
// ============================================================

// Guardar y eliminar la macro DEFAULT 
#ifdef DEFAULT
  #undef DEFAULT
#endif

#include "SpeakerNetwork.h"

// IMPORTANTE: Header con TFLITE_SCHEMA_VERSION
#include "tensorflow/lite/version.h"

#include "modelo_hablante.h"
#include <Arduino.h>
#include <esp_heap_caps.h>

const size_t kArenaSize = 50000;
constexpr int kInputFeatureCount = 80;

// Buffer global para input float
static float input_buffer_float[kInputFeatureCount];

SpeakerNetwork::SpeakerNetwork() : 
    model(nullptr), interpreter(nullptr), input(nullptr), 
    output(nullptr), tensor_arena(nullptr), arena_size(kArenaSize), 
    aligned_arena(nullptr), num_classes(0), isQuantized(false),
    input_scale(1.0f), input_zero_point(0),
    output_scale(1.0f), output_zero_point(0) {}

SpeakerNetwork::~SpeakerNetwork() {
    if (tensor_arena) heap_caps_free(tensor_arena);
}

bool SpeakerNetwork::begin() {
    if (!psramFound()) {
        Serial.println("ERROR: PSRAM no detectada");
        return false;
    }

    model = tflite::GetModel(modelo_hablante_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: Version de modelo incompatible");
        return false;
    }

    resolver_instance.AddFullyConnected();
    resolver_instance.AddSoftmax();
    resolver_instance.AddRelu();
    resolver_instance.AddQuantize();
    resolver_instance.AddDequantize();

    tensor_arena = (uint8_t *)heap_caps_malloc(kArenaSize + 15, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!tensor_arena) {
        Serial.println("ERROR: No se pudo asignar PSRAM");
        return false;
    }

    aligned_arena = (uint8_t *)(((uintptr_t)tensor_arena + 15) & ~15);

    interpreter = new tflite::MicroInterpreter(
        model, resolver_instance, aligned_arena, kArenaSize, &error_reporter_instance);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors fallo");
        return false;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    if (input->bytes < kInputFeatureCount) {
        Serial.printf("ERROR: Input del modelo demasiado pequeño: %d bytes\n", input->bytes);
        return false;
    }
    
    num_classes = output->dims->data[output->dims->size - 1];

    if (input->type == kTfLiteInt8) {
        isQuantized = true;
        input_scale = input->params.scale;
        input_zero_point = input->params.zero_point;
        output_scale = output->params.scale;
        output_zero_point = output->params.zero_point;
        
        Serial.printf("Modelo INT8: %d clases\n", num_classes);
        Serial.printf("Input scale=%.6f zp=%d\n", input_scale, input_zero_point);
        Serial.printf("Output scale=%.6f zp=%d\n", output_scale, output_zero_point);
    } else if (input->type == kTfLiteFloat32) {
        isQuantized = false;
        Serial.printf("Modelo Float32: %d clases\n", num_classes);
    } else {
        Serial.printf("ERROR: Tipo de input no soportado: %d\n", input->type);
        return false;
    }

    return true;
}

float* SpeakerNetwork::getInputBuffer() {
    return input_buffer_float;
}

bool SpeakerNetwork::predict() {
    if (!interpreter) return false;
    
    if (isQuantized) {
        int8_t* int8_input = input->data.int8;
        for (int i = 0; i < kInputFeatureCount; i++) {
            int32_t val = round(input_buffer_float[i] / input_scale) + input_zero_point;
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            int8_input[i] = (int8_t)val;
        }
    } else {
        float* float_input = input->data.f;
        for (int i = 0; i < kInputFeatureCount; i++) {
            float_input[i] = input_buffer_float[i];
        }
    }
    
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("ERROR: Invoke fallo");
        return false;
    }
    
    if (isQuantized) {
        int8_t* int8_output = output->data.int8;
        for (int i = 0; i < num_classes; i++) {
            output_buffer[i] = (int8_output[i] - output_zero_point) * output_scale;
        }
    }
    
    return true;
}

float* SpeakerNetwork::getOutputBuffer() {
    if (isQuantized) {
        return output_buffer;
    } else {
        return output->data.f;
    }
}

int SpeakerNetwork::getNumClasses() {
    return num_classes;
}

void SpeakerNetwork::printMemoryInfo() {
    Serial.printf("RAM Libre: %d KB | PSRAM Libre: %d KB\n", 
                  ESP.getFreeHeap()/1024, ESP.getFreePsram()/1024);
}
