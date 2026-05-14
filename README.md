# MiniXi SpeakerID - ESP32-S3

Sistema de identificación de hablantes usando ESP32-S3 con PSRAM, micrófono I2S, MFCC, Mini Xi-vector y TensorFlow Lite Micro.

## Objetivo

Capturar audio en tiempo real, extraer características de voz y clasificar el hablante directamente en el ESP32-S3 sin depender de una computadora externa.

## Flujo principal

1. Grabar audios por persona.
2. Organizar el dataset en carpetas.
3. Procesar audios en Python.
4. Extraer MFCC.
5. Convertir MFCC a Mini Xi-vector de 80 características.
6. Entrenar un modelo MLP.
7. Exportar a TensorFlow Lite INT8.
8. Integrar el modelo en firmware.
9. Capturar audio en vivo por I2S.
10. Ejecutar inferencia en ESP32.
11. Mostrar el hablante detectado.

## Archivos importantes

- `PIPELINE_COMPLETO.md`: explicación completa del flujo del proyecto.
- `DOCUMENTACION_PROCESO_TRAIN.md`: documentación del procesamiento y entrenamiento.
- `cambios.txt`: resumen de ajustes realizados.
- `src/main.cpp`: flujo principal del firmware.
- `src/MFCC.cpp` y `src/MFCC.h`: extracción MFCC en ESP32.
- `src/SpeakerNetwork.cpp` y `src/SpeakerNetwork.h`: manejo del modelo TensorFlow Lite Micro.
- `src/modelo_hablante.h`: modelo convertido a arreglo C.
- `src/normalizacion.h`: medias y desviaciones usadas para normalizar el Mini Xi-vector.

## Documentación

- [Pipeline completo del sistema](PIPELINE_COMPLETO.md)
- [Documentación del proceso de entrenamiento](DOCUMENTACION_PROCESO_TRAIN.md)
- [Cambios realizados](cambios.txt)
