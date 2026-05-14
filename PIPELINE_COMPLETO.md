# Pipeline completo del proyecto: MiniXi SpeakerID en ESP32-S3 N16R8

## 1. Descripción general del proyecto

Este proyecto implementa un sistema de identificación de hablantes usando un microcontrolador ESP32-S3 con PSRAM. El objetivo principal es que el dispositivo pueda capturar audio en tiempo real mediante un micrófono I2S, procesar la señal, extraer características de voz y clasificar a qué persona pertenece el audio capturado.

El sistema combina procesamiento digital de señales, extracción de características MFCC, una representación compacta tipo Mini Xi-vector y un modelo ligero ejecutado con TensorFlow Lite Micro.

## 2. Pipeline general

Grabación de audios → Organización del dataset → Procesamiento en Python → Segmentación → Filtro de energía → MFCC → Mini Xi-vector → Normalización → Entrenamiento MLP → Evaluación → TensorFlow Lite/INT8 → modelo_hablante.h → Firmware ESP32 → Audio I2S → VAD → MFCC → Mini Xi-vector → Inferencia → Decisión final.

## 3. Dataset

Los audios se organizan por carpetas dentro de dataset_fam. Cada carpeta representa una clase del modelo, por ejemplo hija_1, hijo_1, mama y papa. El notebook detecta automáticamente las carpetas para construir las etiquetas.

## 4. Grabación recomendada

Se recomienda grabar varios audios cortos por persona, por ejemplo 5 audios de 20 segundos. Esto proporciona variedad de frases, tono y energía, evitando que el modelo aprenda segmentos demasiado parecidos de una sola grabación larga.

## 5. Preprocesamiento

Los audios se convierten a mono, se remuestrean a 16 kHz y se normalizan. El punto crítico del proyecto es mantener coherencia entre el audio usado en entrenamiento y el audio capturado en vivo por el ESP32.

## 6. Segmentación

Cada audio se divide en segmentos de 2 segundos. A 16 kHz, cada segmento contiene 32,000 muestras, igual que BUFLEN en el firmware. Se pueden usar ventanas traslapadas para aumentar ejemplos.

## 7. Filtro de energía

Se calcula la energía promedio de cada segmento. Si la energía es menor al umbral, se descarta para evitar entrenar con silencio o ruido débil.

## 8. MFCC

El cálculo MFCC sigue el flujo: preénfasis, ventana Hamming, FFT, espectro de potencia, filtros Mel, logaritmo y DCT. En firmware los parámetros principales son 16 kHz, 2 segundos, ventana 512, salto 256, frecuencia mínima 20 Hz, frecuencia máxima 8 kHz y 20 bandas Mel.

## 9. Recorte central

El firmware calcula una matriz MFCC completa y después toma 63 frames centrales. El resultado usado por el modelo es una matriz de 20 bandas por 63 frames.

## 10. Mini Xi-vector

En lugar de alimentar al modelo con los 1260 valores de la matriz MFCC, se extrae un vector de 80 características. Para cada una de las 20 bandas se calcula media, desviación estándar, máximo y mínimo. Esto produce 20 x 4 = 80 valores.

## 11. Normalización

El vector de 80 características se normaliza usando media y desviación estándar calculadas en Python. Estos parámetros se exportan a normalizacion.h como XVEC_MEAN y XVEC_STD. El ESP32 aplica la misma fórmula: (x - media) / desviación.

## 12. Entrenamiento

El modelo es un MLP ligero con entrada de 80 valores y salida igual al número de hablantes. La arquitectura incluye capas Dense, BatchNormalization, Dropout y una salida softmax para probabilidades por clase.

## 13. Evaluación

Se evalúa con accuracy, reporte de clasificación y matriz de confusión. La matriz permite ver si el modelo confunde a un hablante con otro.

## 14. Exportación

El modelo se guarda en Keras, se convierte a TensorFlow Lite y después a INT8. Finalmente se exporta como modelo_hablante.h para incluirlo en el firmware.

## 15. Firmware ESP32

El firmware se compone principalmente de main.cpp, MFCC.cpp/MFCC.h, SpeakerNetwork.cpp/SpeakerNetwork.h, modelo_hablante.h y normalizacion.h.

## 16. PSRAM

La PSRAM se usa para buffers grandes: audio crudo, audio normalizado, matrices MFCC y tensor arena de TensorFlow Lite Micro. Esto evita saturar la RAM interna del ESP32-S3.

## 17. Captura por I2S

El ESP32 captura audio por I2S a 16 kHz, 32 bits, usando bloques de 32,000 muestras equivalentes a 2 segundos.

## 18. Normalización en vivo

El audio crudo se convierte a float, se filtra con un pasa-altas y se escala para obtener valores entre -1 y 1. Este paso debe coincidir con el procesamiento usado para crear el dataset.

## 19. VAD

Antes de inferir, el sistema valida si hay voz real usando energía, ZCR y varianza. Si detecta silencio, ruido agudo o señal sin variación, descarta el segmento y limpia el historial tras varios frames sin voz.

## 20. Inferencia

Cuando hay voz, el ESP32 calcula MFCC, extrae el Mini Xi-vector, lo normaliza y lo copia al buffer de entrada del modelo. TensorFlow Lite Micro ejecuta la inferencia y devuelve probabilidades.

## 21. Decisión final

El firmware promedia varias predicciones usando VOTE_WINDOW, revisa estabilidad del hablante, compara la mejor clase contra la segunda y aplica un umbral de decisión. Si todo se cumple, imprime el hablante detectado.

## 22. Conclusión

El proyecto demuestra un sistema embebido de reconocimiento de hablantes en tiempo real. La parte más importante es que el entrenamiento en Python y la inferencia en ESP32 mantengan el mismo procesamiento: audio, MFCC, Mini Xi-vector, normalización y decisión. El uso de PSRAM y el vector de 80 características hacen viable ejecutar el modelo en ESP32-S3.
