// ============================================================
// main.cpp - FINAL 
// ============================================================

#include "SpeakerNetwork.h"
#include "MFCC.h"
#include "normalizacion.h"

#include <Arduino.h>
#include "driver/i2s.h"
#include <esp_wifi.h>
#include <esp_bt.h>
#include "esp_task_wdt.h"
#include <math.h>

// es importante definir este valor antes de incluir i2s.h para asegurar que se incluyan las definiciones correctas para el número de puertos I2S disponibles en el ESP32-S3. Esto garantiza que el código pueda compilarse correctamente y utilizar el puerto I2S adecuado para la lectura de audio.
#define BUFLEN 32000

// ==============================
// CONFIG
// ==============================

const int NUM_CLASSES = 4; // Número de hablantes a identificar
const int NUM_FRAMES = 63; 

//  VAD
const float UMBRAL_ENERGIA  = 0.012f;     // El umbral de energía ayuda a filtrar segmentos de audio que son demasiado silenciosos para contener voz. Un valor típico podría ser alrededor de 0.01 a 0.02, pero esto puede variar dependiendo del micrófono y el entorno. Ajustar este umbral puede ayudar a reducir los falsos positivos causados por ruidos de fondo muy suaves.

// Ajustamos segun el entorno 0.25 es un valor típico para ruido agudo, pero puede variar dependiendo del micrófono y el entorno  ZCR es los cambioa de signo en la señal, el ruido agudo suele tener muchos cambios de signo, 
//mientras que la voz humana tiene menos. Ajustar este umbral puede ayudar a filtrar ruidos agudos como el viento o el ruido de fondo.
const float UMBRAL_ZCR_HIGH = 0.25f;   
// El umbral de variación ayuda a detectar si la señal es demasiado constante, lo que podría indicar que no hay voz real. 
const float UMBRAL_VAR      = 0.0002f;

//  DECISIÓN
const float UMBRAL_DECISION = 0.6f;  //0.75f;
const float UMBRAL_INSEGURO = 0.20f;

//  PROMEDIO TEMPORAL   e el número de predicciones que se promedian para tomar una decisión. Un valor más alto puede hacer que el sistema sea más estable pero también más lento para reaccionar a cambios en el hablante. Un valor de 3 a 5 suele ser un buen punto de partida, pero puedes ajustarlo según tus necesidades específicas y la dinámica de tu entorno.
#define VOTE_WINDOW 3

// ==============================
// VARIABLES
// ==============================

int32_t* raw_signal_buf;
float* inputAudio_buf;
float** mfcc_mat_buf;
float** mfcc_temp_buf;

SpeakerNetwork *sn;

static const i2s_port_t i2s_num = I2S_NUM_0;

// estabilidad
int stable_count = 0;
int stable_speaker = -1;
int silencio_frames = 0;

//  HISTORIAL DE PROBABILIDADES
// este historial se utiliza para promediar las predicciones recientes y tomar una decisión más estable sobre el hablante activo. Al almacenar las probabilidades de cada clase en un buffer circular, el sistema puede suavizar las fluctuaciones momentáneas en las predicciones y reducir la probabilidad de cambios erráticos en la identificación del hablante debido a ruidos o variaciones temporales en la señal de audio. El tamaño del buffer (VOTE_WINDOW) determina cuántas predicciones recientes se consideran para el promedio, lo que afecta la estabilidad y la capacidad de respuesta del sistema.
float prob_history[VOTE_WINDOW][NUM_CLASSES];
int history_index = 0;
int history_count = 0;

void reset_decision_state() {
    stable_count = 0;
    stable_speaker = -1;
    history_index = 0;
    history_count = 0;

    for(int i=0;i<VOTE_WINDOW;i++)
        for(int j=0;j<NUM_CLASSES;j++)
            prob_history[i][j] = 0.0f;
}

// ==============================
// NORMALIZAR AUDIO
// ==============================
// se usa void normalized para modificar el buffer de entrada directamente, evitando la necesidad de crear un nuevo buffer para almacenar los datos normalizados. Esto es especialmente importante en un entorno con recursos limitados como el ESP32, donde la eficiencia en el uso de memoria es crucial. Al normalizar el audio, se reduce la cantidad de memoria necesaria para procesar la señal de audio, lo que permite que el sistema funcione de manera más eficiente y con menos riesgo de quedarse sin memoria.
void audio_normalized(int32_t *raw_signal, float *normalized_signal) {
    static float last_filtered = 0.0f;
    static float last_raw = 0.0f;

    for(int j = 0; j < BUFLEN; j++) {
        int32_t raw = raw_signal[j] & 0xFFFFFF00;
        float current_raw = (float)(raw >> 14);
        float filtered = 0.995f * (last_filtered + current_raw - last_raw);

        last_raw = current_raw;
        last_filtered = filtered;

        float val = (filtered * 2.0f) / 32768.0f;

        if(val > 1.0f) val = 1.0f;
        if(val < -1.0f) val = -1.0f;

        normalized_signal[j] = val;
    }
}

// ==============================
// Xi-VECTOR
// ==============================
// La función extract_xvector_features calcula las características del xi-vector a partir de la matriz de MFCC. Para cada banda de Mel, se calculan cuatro características: la media, la desviación estándar, el valor máximo y el valor mínimo. Estas características se almacenan en un vector de salida que se utiliza como entrada para la red neuronal de identificación de hablantes.
void extract_xvector_features(float** mfcc_mat, float* xvec_out) {
    int idx = 0;

    for(int b=0;b<MEL_BANDS;b++){
        float sum=0, sum_sq=0, maxv=-9999, minv=9999;

        for(int t=0;t<NUM_FRAMES;t++){
            float v = mfcc_mat[b][t];
            sum += v;
            sum_sq += v*v;
            if(v>maxv) maxv=v;
            if(v<minv) minv=v;
        }

        float mean = sum/NUM_FRAMES;
        float var = (sum_sq/NUM_FRAMES)-(mean*mean);
        float std = sqrtf(var+1e-9f);

        xvec_out[idx++] = mean;
        xvec_out[idx++] = std;
        xvec_out[idx++] = maxv;
        xvec_out[idx++] = minv;
    }
}

void normalize_xvector(float* xvec){
    for(int i=0;i<XVEC_DIM;i++)
        xvec[i] = (xvec[i]-XVEC_MEAN[i])/XVEC_STD[i];
}

// ==============================
// SETUP
// ==============================

void setup(){
    Serial.begin(115200);
    delay(2000);

    Serial.println("Sistema FINAL - Detección de Hablantes");

    esp_task_wdt_init(60,false);
    esp_task_wdt_add(NULL);

    esp_bt_controller_disable();
    esp_wifi_stop();

    sn = new SpeakerNetwork();
    if(!sn->begin()){
        Serial.println("Error modelo");
        while(1);
    }

    raw_signal_buf = (int32_t*)heap_caps_malloc(BUFLEN*sizeof(int32_t), MALLOC_CAP_SPIRAM);
    inputAudio_buf = (float*)heap_caps_malloc(BUFLEN*sizeof(float), MALLOC_CAP_SPIRAM);

    mfcc_mat_buf = (float**)heap_caps_malloc(MEL_BANDS*sizeof(float*), MALLOC_CAP_SPIRAM);
    mfcc_temp_buf = (float**)heap_caps_malloc(MEL_BANDS*sizeof(float*), MALLOC_CAP_SPIRAM);

    if(!raw_signal_buf || !inputAudio_buf || !mfcc_mat_buf || !mfcc_temp_buf){
        Serial.println("ERROR: No se pudo reservar memoria principal en PSRAM");
        while(1);
    }

    for(int i=0;i<MEL_BANDS;i++){
        mfcc_mat_buf[i] = (float*)heap_caps_malloc(NUM_FRAMES*sizeof(float), MALLOC_CAP_SPIRAM);
        mfcc_temp_buf[i] = (float*)heap_caps_malloc(NUMBER_OF_WINDOWS*sizeof(float), MALLOC_CAP_SPIRAM);

        if(!mfcc_mat_buf[i] || !mfcc_temp_buf[i]){
            Serial.println("ERROR: No se pudo reservar memoria MFCC en PSRAM");
            while(1);
        }
    }

    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .dma_buf_count = 8,
        .dma_buf_len = 512
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = 2,
        .ws_io_num = 7,
        .data_in_num = 1
    };

    i2s_driver_install(i2s_num, &i2s_config, 0, NULL);
    i2s_set_pin(i2s_num, &pin_config);
    i2s_set_clk(i2s_num, 16000, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO);

    Serial.println(">>> Listo");
}

// ==============================
// LOOP
// ==============================

void loop(){
    esp_task_wdt_reset();

    size_t bytes_read;

    if(i2s_read(i2s_num, raw_signal_buf, BUFLEN*sizeof(int32_t), &bytes_read, portMAX_DELAY)==ESP_OK){

        audio_normalized(raw_signal_buf, inputAudio_buf);

        // =========================
        //  VAD
        // =========================

        float energia=0, zcr=0, variance=0;

        for(int i=1;i<BUFLEN;i++){
            energia += fabsf(inputAudio_buf[i]);

            if((inputAudio_buf[i]>0 && inputAudio_buf[i-1]<0) ||
               (inputAudio_buf[i]<0 && inputAudio_buf[i-1]>0))
                zcr++;
        }

        energia /= BUFLEN;
        zcr /= BUFLEN;

        float mean = 0;
        for(int i=0;i<BUFLEN;i++) mean += inputAudio_buf[i];
        mean /= BUFLEN;

        for(int i=0;i<BUFLEN;i++){
            float d = inputAudio_buf[i] - mean;
            variance += d*d;
        }
        variance /= BUFLEN;

        bool silencio = energia < UMBRAL_ENERGIA;
        bool ruido_agudo = zcr > UMBRAL_ZCR_HIGH;
        bool sin_variacion = variance < UMBRAL_VAR;

        if(silencio || ruido_agudo || sin_variacion){
            silencio_frames++;

            if(silencio_frames > 3){
                reset_decision_state();
                Serial.println("🔇 No voz real");
            }
            return;
        }

        silencio_frames = 0;

        // =========================
        // MFCC
        // =========================

        mfccs(inputAudio_buf, mfcc_temp_buf);

        int start = (NUMBER_OF_WINDOWS - NUM_FRAMES)/2;

        for(int i=0;i<MEL_BANDS;i++)
            for(int j=0;j<NUM_FRAMES;j++)
                mfcc_mat_buf[i][j] = mfcc_temp_buf[i][j+start];

        float xvec[XVEC_DIM];
        extract_xvector_features(mfcc_mat_buf,xvec);
        normalize_xvector(xvec);

        float* inBuf = sn->getInputBuffer();
        for(int i=0;i<XVEC_DIM;i++) inBuf[i]=xvec[i];

        if(sn->predict()){

            float* res = sn->getOutputBuffer();

            //  GUARDAR HISTORIAL
            for(int i=0;i<NUM_CLASSES;i++)
                prob_history[history_index][i] = res[i];

            history_index = (history_index + 1) % VOTE_WINDOW;
            if(history_count < VOTE_WINDOW) history_count++;

            //  PROMEDIO
            float avg[NUM_CLASSES] = {0};

            for(int i=0;i<history_count;i++)
                for(int j=0;j<NUM_CLASSES;j++)
                    avg[j] += prob_history[i][j];

            for(int j=0;j<NUM_CLASSES;j++)
                avg[j] /= history_count;

            Serial.print("AVG: ");
            for(int i=0;i<NUM_CLASSES;i++)
                Serial.printf("[%d:%.2f] ", i, avg[i]);
            Serial.println();

            int best=0;
            for(int i=1;i<NUM_CLASSES;i++)
                if(avg[i]>avg[best]) best=i;

            float second=0;
            for(int i=0;i<NUM_CLASSES;i++)
                if(i!=best && avg[i]>second) second=avg[i];

            if(best == stable_speaker) stable_count++;
            else{
                stable_speaker = best;
                stable_count = 0;
            }

            if(history_count < VOTE_WINDOW || stable_count < 2){
                Serial.println("⏳ Analizando...");
                return;
            }

            if((avg[best]-second) < UMBRAL_INSEGURO){
                Serial.println("??? SISTEMA INSEGURO");
                return;
            }

            if(avg[best] > UMBRAL_DECISION){
                const char* nombres[4]={"Hija","Hijo","Mama","Papa"};
                Serial.printf("🎤 HABLANTE: %s (%.1f%%)\n",
                              nombres[best], avg[best]*100);
            }
        }
    }
}
