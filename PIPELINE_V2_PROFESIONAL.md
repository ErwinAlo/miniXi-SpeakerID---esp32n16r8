# Pipeline V2 Profesional - Speaker ID en ESP32-S3 N16R8

## 1. Resumen del sistema

Este proyecto implementa un sistema embebido de identificacion de hablantes en tiempo real sobre un ESP32-S3 N16R8. El modelo usado es hibrido: combina procesamiento digital de senales, extraccion de caracteristicas tipo Xi-vector y una red neuronal MLP ligera. En lugar de enviar audio crudo directamente a la red, el sistema transforma la voz en una representacion compacta de 80 caracteristicas estadisticas derivadas de MFCC. Esa representacion se usa como entrada del clasificador neuronal.

La arquitectura hibrida esta formada por tres niveles:

1. Etapa DSP: captura I2S, normalizacion, VAD, preenfasis, ventana Hamming, FFT, filtros Mel y DCT.
2. Etapa Xi-vector compacta: resumen estadistico de los MFCC mediante media, desviacion estandar, maximo y minimo por banda Mel.
3. Etapa MLP: red neuronal densa que clasifica el vector de 80 caracteristicas y produce probabilidades por hablante.

El sistema captura audio desde un microfono I2S, filtra segmentos sin voz, extrae caracteristicas MFCC, las resume en un Mini Xi-vector de 80 valores y ejecuta un modelo MLP cuantizado con TensorFlow Lite Micro.

El objetivo principal es reconocer el hablante directamente en el microcontrolador, sin depender de una computadora externa durante la inferencia.

Flujo general:

```text
Audio WAV -> dataset_fam -> Python/Notebook -> MFCC -> Mini Xi-vector
-> Normalizacion -> MLP -> TFLite INT8 -> modelo_hablante.h
-> ESP32-S3 -> I2S -> VAD -> MFCC -> Mini Xi-vector
-> Normalizacion -> Inferencia -> Promedio temporal -> Hablante detectado
```

Diagrama de arquitectura:

![Diagrama de arquitectura MiniXi-vector + MLP](diagramaXi-vec.png)

### 1.1 Revision del diagrama

El diagrama es correcto como representacion conceptual del sistema V2. Coincide con el firmware y el entrenamiento en los puntos principales:

1. Entrada de audio de 16 kHz, 2 segundos y 32,000 muestras.
2. Extraccion MFCC mediante preenfasis, ventana Hamming, FFT, banco Mel, log y DCT.
3. Matriz MFCC de `20 x 63` frames centrales.
4. Extraccion Mini Xi-vector con media, desviacion estandar, maximo y minimo por banda Mel.
5. Vector final de `80` caracteristicas.
6. Normalizacion tipo StandardScaler usando media y desviacion del conjunto de entrenamiento.
7. Clasificador MLP con capas densas `128 -> 64 -> 32 -> num_classes`.
8. Salida Softmax y decision final con estados como sin voz, analizando, inseguro o nombre del hablante.

Observaciones importantes:

1. La franja amarilla `Xi-Vector MLP - Identificacion de Hablante` debe leerse como el nombre de la arquitectura completa, no como una capa adicional entre el audio y los MFCC.
2. Los valores de Dropout del diagrama (`0.30`, `0.30` y `0.20`) coinciden con `Train/procesamiento_log.txt` y con la arquitectura documentada del modelo.
3. Los tiempos al pie del diagrama solo deben mantenerse si fueron medidos desde el Monitor Serial del ESP32-S3 real. El firmware imprime `Tiempo MFCC`, `Tiempo Inferencia`, `Tiempo Total`, RAM y PSRAM, por lo que esos valores deben venir de una corrida real y no de una estimacion.

## 2. Hardware objetivo

El hardware objetivo es un modulo ESP32-S3 N16R8:

| Elemento | Descripcion |
| --- | --- |
| Microcontrolador | ESP32-S3 |
| Flash | 16 MB, indicado por N16 |
| PSRAM | 8 MB, indicado por R8 |
| Frecuencia CPU | Hasta 240 MHz |
| Audio | Microfono digital I2S |
| Muestreo | 16 kHz |
| Formato I2S | 32 bits, mono |
| Framework | Arduino sobre PlatformIO |
| Inferencia | TensorFlow Lite Micro |

En `platformio.ini` el proyecto habilita PSRAM y configura el uso de flash de 16 MB:

```ini
board_build.psram = 1
board_build.arduino.memory_type = qio_opi
board_build.f_flash = 80000000L
board_build.flash_mode = qio
board_upload.flash_size = 16MB
```

La PSRAM es importante porque el pipeline reserva buffers grandes para audio, matrices MFCC, FFT y tensor arena del modelo. El firmware valida su presencia con `psramFound()` y reserva memoria con `heap_caps_malloc(..., MALLOC_CAP_SPIRAM)`.

## 3. Arquitectura funcional

El sistema esta dividido en dos bloques principales:

1. Entrenamiento en Python
2. Inferencia embebida en ESP32-S3

### 3.1 Entrenamiento en Python

El entrenamiento se realiza desde `Train/model.ipynb`. El notebook:

1. Lee los audios organizados por persona en `dataset_fam/`.
2. Convierte el audio a mono y 16 kHz.
3. Divide el audio en segmentos de 2 segundos.
4. Aplica filtro de energia para descartar silencio.
5. Calcula MFCC compatibles con el firmware.
6. Extrae Mini Xi-vectors de 80 valores.
7. Calcula media y desviacion estandar para normalizacion.
8. Entrena un MLP de clasificacion.
9. Exporta el modelo a Keras, TFLite y TFLite INT8.
10. Genera `Train/modelo_hablante.h` para integrarlo en firmware.

### 3.1.1 Dataset usado

El dataset fuente se organiza en carpetas por hablante dentro de `dataset_fam/`. En la version actual del entrenamiento, las clases encontradas son:

```text
hija, hijo, mama, papa
```

Cada persona tiene 8 grabaciones WAV. La estructura de grabacion es:

| Grupo | Cantidad por persona | Contenido | Fuente o intencion |
| --- | ---: | --- | --- |
| Preguntas naturales | 5 | saludo, vacaciones, gustos, molestia de la semana y rutina del dia | Primeras cinco preguntas de `preguntas.txt` |
| Situaciones de discusion | 3 | dialogos o respuestas con mayor tension y variacion emocional | Grabaciones adicionales de discusion |

Las primeras 5 grabaciones buscan que cada persona hable de forma cotidiana, sin leer frases rigidas. Esto ayuda a capturar variaciones naturales de ritmo, pausas, volumen, entonacion y forma de expresarse. Las 3 grabaciones de discusion agregan situaciones con mayor carga emocional, donde pueden aparecer cambios de tono, velocidad, energia y pronunciacion. La combinacion hace que el modelo vea mas variabilidad real por hablante.

`preguntas.txt` tambien propone condiciones de fondo como ventilador, TV de fondo, gente hablando lejos, cocina, calle, eco de cuarto, computadora y aire acondicionado. Estas condiciones son utiles como variacion ambiental durante la captura, siempre que la voz principal siga siendo clara.

Segun `Train/procesamiento_log.txt`, el entrenamiento actual proceso 8 archivos por persona:

| Clase | Archivos WAV | Segmentos finales |
| --- | ---: | ---: |
| `hija` | 8 | 294 |
| `hijo` | 8 | 292 |
| `mama` | 8 | 295 |
| `papa` | 8 | 295 |
| Total | 32 | 1176 |

Cada audio se divide en segmentos de 2 segundos y cada segmento se transforma en un Mini Xi-vector de 80 caracteristicas. El dataset final queda como:

```text
X: 1176 segmentos x 80 caracteristicas
y: 1176 etiquetas
```

Archivos principales generados:

| Archivo | Funcion |
| --- | --- |
| `X_xvector.npy` | Matriz de caracteristicas de entrenamiento |
| `y_labels.npy` | Etiquetas numericas |
| `labels.npy` | Nombres de hablantes |
| `xvector_mean.npy` | Media por caracteristica |
| `xvector_std.npy` | Desviacion estandar por caracteristica |
| `normalizacion_params.txt` | Parametros para `src/normalizacion.h` |
| `xvector_model.h5` | Modelo Keras |
| `xvector_model.tflite` | Modelo TFLite float |
| `xvector_model_int8.tflite` | Modelo TFLite cuantizado INT8 |
| `modelo_hablante.h` | Modelo convertido a arreglo C |
| `confusion_matrix_xvector.png` | Matriz de confusion |

### 3.2 Inferencia embebida

La inferencia ocurre en `src/main.cpp` y `src/SpeakerNetwork.cpp`.

El firmware:

1. Inicializa Serial, watchdog, PSRAM, modelo TFLite Micro e I2S.
2. Captura 2 segundos de audio a 16 kHz.
3. Normaliza el audio crudo del microfono.
4. Aplica VAD con energia, ZCR y varianza.
5. Calcula MFCC.
6. Selecciona 63 frames centrales.
7. Extrae un Mini Xi-vector de 80 caracteristicas.
8. Normaliza el vector con `XVEC_MEAN` y `XVEC_STD`.
9. Copia el vector al input del modelo.
10. Ejecuta `interpreter->Invoke()`.
11. Dequantiza la salida INT8 a probabilidades.
12. Promedia predicciones recientes con una ventana temporal de 3 resultados.
13. Decide el hablante si supera umbrales de confianza y estabilidad.

### 3.3 Arquitectura hibrida del modelo con caracteristicas de Xi-vector

La arquitectura no es solamente una red neuronal. Es un pipeline hibrido donde una parte del trabajo la realiza el procesamiento de senales y otra parte la aprende el modelo.

```text
Audio 2 s / 16 kHz
-> VAD
-> MFCC 20 x 63
-> Mini Xi-vector 80
-> Normalizacion estadistica
-> MLP cuantizado INT8
-> Softmax
-> Hablante detectado
```

Esta separacion tiene una razon practica: el ESP32-S3 no necesita procesar una senal enorme dentro de una red profunda. Primero se extrae informacion acustica compacta y despues el MLP aprende la frontera de decision entre hablantes.

Responsabilidad de cada bloque:

| Bloque | Entrada | Salida | Responsabilidad |
| --- | --- | --- | --- |
| Captura I2S | Audio digital 32 bits | Buffer de 32,000 muestras | Obtener 2 segundos de audio |
| VAD | Audio normalizado | Aceptar o rechazar segmento | Evitar silencio y ruido no util |
| MFCC | Audio valido | Matriz `20 x 63` | Extraer forma espectral de la voz |
| Mini Xi-vector | MFCC | Vector de 80 valores | Compactar la informacion temporal |
| Normalizacion | Vector de 80 | Vector escalado | Igualar distribucion entrenamiento-firmware |
| MLP | Vector normalizado | Probabilidades | Clasificar hablante |
| Decision temporal | Probabilidades recientes | Hablante final | Suavizar predicciones inestables |

## 4. Procesamiento de audio

El audio se captura con I2S:

| Parametro | Valor |
| --- | --- |
| Frecuencia de muestreo | 16 kHz |
| Duracion por bloque | 2 segundos |
| Muestras por bloque | 32,000 |
| Bits por muestra I2S | 32 bits |
| Canal | Mono |
| Buffer principal | `BUFLEN = 32000` |

El firmware aplica una normalizacion inicial en `audio_normalized()`. Esta etapa convierte la senal cruda I2S a valores flotantes en el rango aproximado `[-1, 1]`, aplica filtrado de componente DC y limita saturaciones.

## 5. VAD: deteccion de voz valida

Antes de ejecutar MFCC e inferencia, el firmware descarta segmentos que probablemente no contienen voz util. Esto evita gastar tiempo de CPU con silencio, ruido agudo o senales demasiado constantes.

Metricas usadas:

| Metrica | Umbral | Funcion |
| --- | --- | --- |
| Energia media | `UMBRAL_ENERGIA = 0.012f` | Descarta silencio o senales muy debiles |
| ZCR | `UMBRAL_ZCR_HIGH = 0.25f` | Descarta ruido agudo con muchos cruces por cero |
| Varianza | `UMBRAL_VAR = 0.0002f` | Descarta senales demasiado constantes |

Si el segmento falla el VAD durante varios ciclos, el sistema reinicia el historial de decision para evitar arrastrar predicciones antiguas.

## 6. MFCC

Los MFCC son la representacion espectral base del sistema. El firmware y el notebook mantienen el mismo procesamiento para que el modelo reciba en el ESP32 datos con la misma forma que vio durante el entrenamiento.

Parametros principales:

| Parametro | Valor |
| --- | --- |
| Preenfasis | `0.97` |
| Ventana | Hamming |
| Tamano de ventana | 512 muestras |
| Salto | 256 muestras |
| Bandas Mel | 20 |
| Frecuencia minima | 20 Hz |
| Frecuencia maxima | 8 kHz |
| Frames usados por modelo | 63 centrales |

Flujo MFCC:

```text
Audio normalizado
-> preenfasis
-> ventana Hamming
-> FFT real
-> espectro de potencia
-> filtros triangulares Mel
-> log(mel + 1e-6)
-> DCT
-> matriz MFCC de 20 bandas x 63 frames
```

## 7. Que es el Mini Xi-vector

En este proyecto, el Xi-vector no es un x-vector clasico usado en sistemas grandes de speaker recognition. Aqui se usa una version compacta y embebible: un vector estadistico derivado de MFCC.

La matriz MFCC tiene 20 bandas y 63 frames. Para cada banda se calculan cuatro estadisticas:

1. Media
2. Desviacion estandar
3. Maximo
4. Minimo

Entonces:

```text
20 bandas Mel x 4 estadisticas = 80 caracteristicas
```

Ese vector de 80 valores resume el comportamiento espectral del hablante en una ventana de audio de 2 segundos. Es mucho mas pequeno que alimentar todos los MFCC directamente al modelo:

```text
MFCC completo: 20 x 63 = 1260 valores
Mini Xi-vector: 80 valores
```

Esto reduce memoria, computo y tamano del modelo, que son tres factores criticos en un ESP32-S3.

## 8. Que es un MLP de Xi-vector

Un MLP, o Multi-Layer Perceptron, es una red neuronal feed-forward compuesta por capas densas. En este proyecto, el MLP no procesa audio crudo; procesa el Mini Xi-vector de 80 caracteristicas.

La idea es:

```text
Mini Xi-vector normalizado -> MLP -> probabilidades por hablante
```

El MLP aprende relaciones entre las estadisticas espectrales del audio y la identidad del hablante. Como la entrada ya esta resumida, la red puede ser pequena y apta para microcontrolador.

### 8.1 Entrada del MLP

La entrada del MLP es un vector de dimension fija:

```text
X = [mean_0, std_0, max_0, min_0, ..., mean_19, std_19, max_19, min_19]
```

Cada grupo de cuatro valores describe una banda Mel. Como hay 20 bandas:

```text
20 bandas x 4 estadisticas = 80 entradas
```

Antes de entrar al modelo, cada valor se normaliza con los parametros calculados durante entrenamiento:

```text
x_norm = (x - XVEC_MEAN) / XVEC_STD
```

Esto es critico: el MLP fue entrenado con vectores normalizados, por lo que el firmware debe aplicar exactamente la misma normalizacion para que la inferencia sea coherente.

### 8.2 Capas internas

Arquitectura del modelo entrenado:

| Capa | Salida | Funcion |
| --- | --- | --- |
| Input | 80 | Mini Xi-vector normalizado |
| Dense + ReLU | 128 | Aprende combinaciones no lineales de las caracteristicas |
| BatchNormalization | 128 | Estabiliza la distribucion interna |
| Dropout | 128 | Reduce sobreajuste |
| Dense + ReLU `xvector_embedding` | 64 | Capa intermedia de representacion |
| BatchNormalization | 64 | Estabiliza entrenamiento |
| Dropout | 64 | Reduce sobreajuste |
| Dense + ReLU | 32 | Compacta la decision |
| Dropout | 32 | Regularizacion final |
| Dense + Softmax | `num_classes` | Probabilidad por hablante |

El flujo interno puede leerse asi:

```text
80 caracteristicas
-> Dense(128, ReLU)
-> BatchNormalization
-> Dropout(0.3)
-> Dense(64, ReLU, xvector_embedding)
-> BatchNormalization
-> Dropout(0.3)
-> Dense(32, ReLU)
-> Dropout(0.2)
-> Dense(num_classes, Softmax)
```

### 8.3 Funcion de cada bloque del MLP

La primera capa `Dense(128)` expande las 80 caracteristicas de entrada a un espacio de mayor dimension. Esto permite que la red combine relaciones entre bandas Mel, por ejemplo diferencias de energia, forma espectral y variaciones estadisticas que pueden distinguir una voz de otra.

Las capas `BatchNormalization` ayudan a estabilizar el entrenamiento. Mantienen las activaciones internas en rangos mas controlados, lo que permite que la red aprenda de forma mas consistente.

Las capas `Dropout` se usan solo durante entrenamiento. Apagan aleatoriamente parte de las activaciones para reducir sobreajuste. Esto obliga al modelo a no depender de una sola combinacion de caracteristicas y mejora la generalizacion cuando escucha audio nuevo.

La capa `Dense(64, name="xvector_embedding")` actua como representacion intermedia. No reemplaza el Mini Xi-vector de entrada; mas bien aprende una representacion neuronal mas compacta a partir de esas 80 caracteristicas. Por eso el sistema puede describirse como hibrido: primero usa un Xi-vector estadistico y luego una red aprende una representacion discriminativa.

La capa `Dense(32)` comprime la informacion antes de la salida. Funciona como una etapa final de decision que prepara las activaciones para la clasificacion.

La capa `Dense(num_classes, Softmax)` produce una probabilidad por hablante. En el entrenamiento actual `num_classes = 4`, por lo que la salida tiene cuatro probabilidades en el mismo orden que `labels.npy` y `src/speaker_labels.h`.

### 8.4 Tamano del modelo

El MLP es pequeno comparado con redes de audio mas profundas. Su tamano es adecuado para el ESP32-S3 porque usa una entrada compacta y solo capas densas.

| Bloque | Parametros aproximados | Comentario |
| --- | --- | --- |
| Dense 80 -> 128 | 10,368 | Primera expansion de caracteristicas |
| Dense 128 -> 64 | 8,256 | Representacion `xvector_embedding` |
| Dense 64 -> 32 | 2,080 | Compresion previa a salida |
| Dense 32 -> 4 | 132 | Clasificador final |
| BatchNormalization | Incluido en el total | Parametros de escala, sesgo y estadisticas |
| Dropout | 0 | Solo regulariza durante entrenamiento |

El modelo actual reporta `21,604` parametros totales. Despues se convierte a TensorFlow Lite y se cuantiza a INT8 para reducir el costo de memoria e inferencia en el microcontrolador.

### 8.5 Por que no se usa audio crudo

Procesar audio crudo directamente implicaria entregar al modelo 32,000 muestras por ventana de 2 segundos. Eso aumentaria mucho el tamano de la red, el consumo de memoria y el tiempo de inferencia.

Con la arquitectura hibrida:

```text
32,000 muestras de audio -> 1,260 valores MFCC -> 80 valores Xi-vector -> MLP pequeno
```

El modelo conserva informacion relevante para distinguir hablantes, pero evita cargar al ESP32-S3 con una red profunda de audio. Esta es la razon principal para usar un MLP de Xi-vector en lugar de una CNN, RNN o red entrenada directamente sobre waveform.

El entrenamiento actual reporta:

| Metrica | Valor |
| --- | --- |
| Clases | `hija`, `hijo`, `mama`, `papa` |
| Dataset procesado | 1176 segmentos x 80 caracteristicas |
| Parametros del modelo | 21,604 |
| Accuracy de validacion | 0.9831 |
| Accuracy reporte final | 0.98 sobre 236 muestras |

## 9. Cuantizacion INT8

El modelo se exporta a TensorFlow Lite INT8 para reducir tamano y mejorar viabilidad en el ESP32-S3.

Ventajas:

1. Menor tamano en flash.
2. Menor uso de memoria.
3. Operaciones mas ligeras que Float32.
4. Mejor integracion con TensorFlow Lite Micro.

En `SpeakerNetwork.cpp`, el firmware detecta si el input del modelo es `kTfLiteInt8`. Si es cuantizado:

1. Convierte cada entrada float normalizada a int8 usando `input_scale` y `input_zero_point`.
2. Ejecuta `interpreter->Invoke()`.
3. Convierte la salida int8 de regreso a probabilidades float con `output_scale` y `output_zero_point`.

## 10. Memoria y PSRAM

La arquitectura usa PSRAM para alojar estructuras grandes:

| Recurso | Ubicacion |
| --- | --- |
| Audio crudo I2S | PSRAM |
| Audio normalizado | PSRAM |
| Matriz MFCC final | PSRAM |
| Matriz MFCC temporal | PSRAM |
| Matriz de ventana | PSRAM |
| Matriz FFT | PSRAM |
| Filtros Mel | PSRAM |
| Tensor arena TFLite Micro | PSRAM |

El `tensor_arena` esta configurado con:

```cpp
const size_t kArenaSize = 50000;
```

Ademas, el firmware imprime memoria libre despues de cada inferencia:

```text
RAM Libre: <valor> KB | PSRAM Libre: <valor> KB
```

Esto permite validar si el sistema sigue estable durante ejecucion continua.

## 11. Decision temporal

El modelo entrega probabilidades por clase, pero el firmware no decide solo con una prediccion aislada. Para mejorar estabilidad se usa una ventana temporal:

```cpp
#define VOTE_WINDOW 3
```

El sistema promedia las ultimas 3 predicciones y aplica:

| Parametro | Valor | Funcion |
| --- | --- | --- |
| `UMBRAL_DECISION` | `0.6f` | Confianza minima para aceptar hablante |
| `UMBRAL_INSEGURO` | `0.20f` | Diferencia minima entre la mejor clase y la segunda |
| `stable_count` | 2 ciclos | Evita cambios bruscos de hablante |

Esto hace que la salida sea mas estable frente a ruido, variacion de voz o predicciones momentaneamente ambiguas.

## 12. Tiempos de ejecucion

Si, los tiempos de ejecucion deben mencionarse en una documentacion profesional, porque muestran si el sistema cumple con restricciones de tiempo real.

El firmware ya mide tres tiempos con `micros()`:

```cpp
Serial.printf("Tiempo MFCC: %.2f ms\n", elapsed_ms(mfcc_start_us, mfcc_end_us));
Serial.printf("Tiempo Inferencia: %.2f ms\n", elapsed_ms(inference_start_us, inference_end_us));
Serial.printf("Tiempo Total: %.2f ms\n", elapsed_ms(total_start_us, inference_end_us));
```

Interpretacion:

| Tiempo | Que mide |
| --- | --- |
| Tiempo MFCC | Extraccion MFCC, seleccion de frames, Mini Xi-vector y normalizacion |
| Tiempo Inferencia | Solo ejecucion del modelo TFLite Micro |
| Tiempo Total | Pipeline desde inicio de procesamiento hasta salida del modelo |

Tabla recomendada para reporte final:

| Metrica | Valor medido | Fuente |
| --- | --- | --- |
| Tiempo MFCC | Pendiente de medicion serial | Monitor Serial |
| Tiempo Inferencia | Pendiente de medicion serial | Monitor Serial |
| Tiempo Total | Pendiente de medicion serial | Monitor Serial |
| RAM libre | Pendiente de medicion serial | `sn->printMemoryInfo()` |
| PSRAM libre | Pendiente de medicion serial | `sn->printMemoryInfo()` |

No conviene inventar estos valores. Lo correcto es capturarlos desde el Monitor Serial en el ESP32-S3 N16R8 real, porque dependen de la version del framework, optimizaciones, velocidad de flash, PSRAM, compilacion y carga del sistema.

## 13. Archivos de firmware

| Archivo | Responsabilidad |
| --- | --- |
| `src/main.cpp` | Captura I2S, VAD, MFCC, Mini Xi-vector, normalizacion, decision |
| `src/MFCC.cpp` | Implementacion de MFCC compatible con entrenamiento |
| `src/MFCC.h` | Constantes de audio, ventana, FFT y Mel |
| `src/SpeakerNetwork.cpp` | Inicializacion e inferencia TFLite Micro |
| `src/SpeakerNetwork.h` | Interfaz de la red neuronal |
| `src/modelo_hablante.h` | Modelo TFLite INT8 embebido |
| `src/normalizacion.h` | Media y desviacion estandar del Xi-vector |
| `src/speaker_labels.h` | Etiquetas generadas desde `Train/labels.npy` |

## 14. Pipeline de reentrenamiento

Cuando se agregan nuevos audios o hablantes:

1. Organizar audios por persona dentro de `dataset_fam/`.
2. Ejecutar completo `Train/model.ipynb`.
3. Revisar accuracy, matriz de confusion y reporte de clasificacion.
4. Copiar `Train/modelo_hablante.h` hacia `src/modelo_hablante.h`.
5. Actualizar `src/normalizacion.h` con los valores de `Train/normalizacion_params.txt`.
6. Compilar con PlatformIO.
7. Verificar que `Train/generate_speaker_labels.py` regenere `src/speaker_labels.h`.
8. Subir firmware al ESP32-S3 N16R8.
9. Capturar tiempos de MFCC, inferencia, total, RAM y PSRAM desde Monitor Serial.

## 15. Criterios de validacion

Para considerar valida una nueva version del modelo:

1. La cantidad de clases del modelo debe coincidir con `NUM_CLASSES`.
2. `speaker_labels.h` debe coincidir con el orden de `labels.npy`.
3. La normalizacion de firmware debe coincidir con la del entrenamiento.
4. El accuracy de validacion debe ser aceptable para el numero de hablantes.
5. La matriz de confusion no debe mostrar confusiones fuertes entre dos personas.
6. El tiempo total debe ser suficientemente menor al intervalo entre bloques de audio.
7. La memoria libre de RAM y PSRAM debe mantenerse estable durante ejecucion continua.

## 16. Conclusion tecnica

La arquitectura V2 combina un procesamiento clasico de audio con una red neuronal ligera. El uso de MFCC reduce el audio a informacion espectral relevante; el Mini Xi-vector resume esa informacion a 80 estadisticas compactas; el MLP aprende a mapear esas estadisticas hacia hablantes; y la cuantizacion INT8 permite ejecutar el modelo en un ESP32-S3 N16R8 con TensorFlow Lite Micro.

El resultado es una arquitectura balanceada para sistemas embebidos: suficientemente ligera para correr en microcontrolador, pero expresiva para identificar hablantes reales usando audio capturado en vivo.
