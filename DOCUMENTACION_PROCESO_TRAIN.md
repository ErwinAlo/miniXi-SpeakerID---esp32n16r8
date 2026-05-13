# Documentación del proceso y entrenamiento

## 1. ¿Qué es este proyecto?
Este proyecto es un identificador de hablantes basado en un microcontrolador ESP32 con PSRAM. Usa audio grabado de 4 clases (`hija_1`, `hijo_1`, `mama`, `papa`) y entrena un modelo ligero que se ejecuta en el firmware.

El pipeline se divide en dos etapas principales:
- `Process`: carga y preprocesa el dataset, genera características tipo Xi-vector.
- `Train`: entrena un modelo MLP pequeño sobre esas características y exporta el modelo a TensorFlow Lite.

## 2. Carga del dataset

El dataset se encuentra en `dataset_fam/` 
Estructura:
- `dataset_fam/hija_1/`
- `dataset_fam/hijo_1/`
- `dataset_fam/mama/`
- `dataset_fam/papa/`


Cada carpeta contiene archivos WAV de audio. El notebook `Train/model.ipynb` carga automáticamente las clases presentes en `dataset_fam` con:
```python
personas = sorted([d for d in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, d))])
```

Esto permite detectar dinámicamente las clases de hablante sin codificarlas manualmente.

### Versiones con y sin la clase `ruido`

En este proyecto existen dos enfoques de dataset:
- una versión con la clase `ruido` incluida como categoría separada
- una versión sin la clase `ruido`, donde solo se entrenan las clases de hablantes reales

No se usa mezcla de voz y ruido para generar segmentos mixtos. En cambio, el dataset `ruido` se trata como una clase independiente y separada. La razón es que los segmentos mixtos de voz+ruido pueden confundir al modelo: dificultan que aprenda una representación estable del hablante y mezclan dos distribuciones muy diferentes de características.

Es mejor que el modelo primero distinga claramente "voz real" de "ruido" y luego, si se desea, aplicar una segunda etapa de robustez al ruido. Esto también simplifica el entrenamiento y hace más consistente la normalización, porque los segmentos de ruido puros tienen una distribución muy diferente a la de la voz hablada.

## 3. Segmentación

La segmentación se hace en el notebook del proceso con:
- duración de segmento: 2 segundos
- salto entre segmentos: 0.5 segundos

Código equivalente:
```python
for i in range(0, len(audio_proc) - samples_segment + 1, step_segment):
    seg = audio_proc[i:i + samples_segment]
```

Esto genera ventanas superpuestas del audio, lo que aumenta la cantidad de ejemplos y permite capturar distintas partes del habla.

## 4. Filtro de energía

Antes de procesar cada segmento, se calcula la energía media:
```python
energia = np.mean(np.abs(seg))
if energia < energia_umbral:
    continue
```

El umbral usado en el notebook es:
- `energia_umbral = 0.0025`

Esto elimina segmentos que contienen silencio o ruido muy débil.

### En el firmware

En `src/main.cpp` también hay validación de voz (VAD) con tres métricas:
- energía (`UMBRAL_ENERGIA = 0.012f`)
- tasa de cruces por cero / ZCR (`UMBRAL_ZCR_HIGH = 0.25f`)
- varianza (`UMBRAL_VAR = 0.0002f`)

Si el segmento cumple cualquiera de estas condiciones, se descarta y no se procesa.

## 5. MFCC: extracción de características

El cálculo de MFCC en el notebook está diseñado para coincidir con el firmware ESP32.
En `Train/model.ipynb` y `src/MFCC.cpp` se siguen los mismos pasos:
- preénfasis con coeficiente `0.97`
- ventana Hamming
- FFT real
- espectro de potencia (`abs(fft)**2` en Python, equivalente a `r*r + im*im` en C)
- filtros triangulares Mel con `Mel_bands = 20`
- logaritmo `log(mel + 1e-6)`
- DCT ortogonal (`dct(..., norm='ortho')`)

El resultado es una matriz MFCC de tamaño:
- `20` bandas Mel
- `63` frames centrales

En el firmware, después de calcular MFCC con `mfccs()`, se extrae el bloque central con 63 frames:
```cpp
int start = (NUMBER_OF_WINDOWS - NUM_FRAMES)/2;
for(int i=0;i<MEL_BANDS;i++)
    for(int j=0;j<NUM_FRAMES;j++)
        mfcc_mat_buf[i][j] = mfcc_temp_buf[i][j+start];
```

## 6. Arquitectura híbrida con características tipo Xi-vector

### ¿Qué significa "Xi-vector" en este proyecto?
No es un x-vector clásico de redes profundas de speaker recognition, sino una representación compacta basada en estadísticas de MFCC.

La idea es resumir cada banda MFCC con cuatro valores:
- media
- desviación estándar
- máximo
- mínimo

Esto produce un vector de características de `20 bandas * 4 = 80` valores.

En el notebook:
```python
for band in range(Mel_bands):
    band_data = mfcc[band, :]
    features.append(np.mean(band_data))
    features.append(np.std(band_data))
    features.append(np.max(band_data))
    features.append(np.min(band_data))
```

Y en el firmware `src/main.cpp`:
```cpp
void extract_xvector_features(float** mfcc_mat, float* xvec_out) {
    for(int b=0;b<MEL_BANDS;b++){
        float sum=0, sum_sq=0, maxv=-9999, minv=9999;
        for(int t=0;t<NUM_FRAMES;t++){
            float v = mfcc_mat[b][t];
            ...
        }
        xvec_out[idx++] = mean;
        xvec_out[idx++] = std;
        xvec_out[idx++] = maxv;
        xvec_out[idx++] = minv;
    }
}
```

## 7. Normalización

Después de generar el vector de 80 características, se normaliza con media y desviación estándar calculadas en el proceso de entrenamiento.

En `src/normalizacion.h` se definen:
- `XVEC_MEAN[80]`
- `XVEC_STD[80]`

Y en `src/main.cpp`:
```cpp
for(int i=0;i<XVEC_DIM;i++)
    xvec[i] = (xvec[i]-XVEC_MEAN[i])/XVEC_STD[i];
```

Esto asegura que el modelo reciba datos con la misma distribución que el entrenamiento.

## 8. Entrenamiento del modelo

### Datos usados
- `X_xvector.npy`: matriz de características normalizadas de forma estándar
- `y_labels.npy`: etiquetas numéricas de clases
- `labels.npy`: nombres de clases

El dataset se divide en:
- 80% entrenamiento
- 20% validación

También se calcula `class_weights` para balancear el entrenamiento si hay diferencias de cantidad entre clases.

### Arquitectura del modelo

El modelo entrenado en `Train/model.ipynb` es un pequeño MLP con las siguientes capas:
- `Dense(128, relu)`
- `BatchNormalization`
- `Dropout(0.3)`
- `Dense(64, relu, name='xvector_embedding')`
- `BatchNormalization`
- `Dropout(0.3)`
- `Dense(32, relu)`
- `Dropout(0.2)`
- `Dense(num_classes, softmax)`

Esta arquitectura es la que en el notebook se llama "Mini Xi-vector".

### Compilación y evaluación

El modelo se compila con:
- pérdida: `sparse_categorical_crossentropy`
- optimizador: `Adam`
- métrica: `accuracy`

Se usan callbacks:
- `EarlyStopping` sobre `val_accuracy`
- `ReduceLROnPlateau` sobre `val_loss`

Para evaluación se genera:
- reporte de clasificación
- matriz de confusión

## 9. Archivos generados

### Resultado del proceso (`Train/model.ipynb`)
- `X_xvector.npy` → características finales (80 valores por segmento)
- `y_labels.npy` → etiquetas numéricas
- `labels.npy` → nombres de clase
- `xvector_mean.npy` → medias de cada característica
- `xvector_std.npy` → desviaciones estándar de cada característica
- `normalizacion_params.txt` → valores que se convierten a `src/normalizacion.h`

### Resultado del entrenamiento
- `xvector_model.h5` → modelo Keras guardado
- `xvector_model.tflite` → modelo TensorFlow Lite float
- `xvector_model_int8.tflite` → modelo TensorFlow Lite INT8 optimizado
- `modelo_hablante.h` → versión C del modelo INT8 para firmware
- `confusion_matrix_xvector.png` → matriz de confusión de validación

### Firmware final
- `src/modelo_hablante.h` → modelo embebido en C
- `src/normalizacion.h` → media y std para normalizar el Xi-vector
- `src/main.cpp` → lógica de adquisición de audio, VAD, MFCC, Xi-vector, normalización y predicción
- `src/MFCC.cpp` / `src/MFCC.h` → cálculo de MFCC compatible con el notebook
- `src/SpeakerNetwork.cpp` / `src/SpeakerNetwork.h` → carga e inferencia del modelo TFLite

## 10. Relación entre `process` y `train`

- `process` transforma audio crudo en vectores Xi-vector normalizados.
- `train` consume esos vectores para aprender a clasificar al hablante.
- Después del entrenamiento, se exporta el modelo a TFLite e INT8, y se genera `modelo_hablante.h` para el firmware.
- Además, `normalizacion_params.txt` se usa para crear la normalización de `XVEC_MEAN` y `XVEC_STD` en firmware.

## 11. Flujo general completo

1. Leer audio WAV de `dataset_fam`
2. Normalizar volumen / preprocesar exactamente como ESP32
3. Dividir audio en segmentos de 2s con salto de 0.5s
4. Filtrar segmentos con baja energía
5. Calcular MFCC iguales a los usados en firmware
6. Reducir MFCC a Xi-vector de 80 valores
7. Normalizar Xi-vector
8. Guardar datos y entrenar modelo
9. Exportar modelo a TFLite/INT8
10. Integrar en firmware con `src/modelo_hablante.h` y `src/normalizacion.h`

---

## 12. Notas importantes

- La validación de voz en firmware no es solo energía, también usa ZCR y varianza para reducir falsos positivos de ruido.
- El enfoque Xi-vector híbrido resume información spectral del hablante sin usar una red de embeddings profunda completa.
- El entrenamiento y el firmware comparten el mismo preprocesamiento de MFCC para mantener coherencia entre los datos de entrenamiento y el audio en vivo.
