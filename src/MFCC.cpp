#include "MFCC.h"
#include "FFT.h"
#include "arduino.h"
#include <math.h>

static fft_config_t *real_fft_plan = nullptr;
static float **win_mat = nullptr;
static float **fft_mat = nullptr;
static float **tri_mat = nullptr;

static bool matrices_initialized = false;

// --------------------------------------
// OPTIMIZACIONES
// --------------------------------------

// Ventana Hamming precalculada
static float window_global[SIZE_WINDOW];
static bool window_ready = false;

// Matriz DCT precalculada
static float dct_matrix[MEL_BANDS][MEL_BANDS];
static bool dct_ready = false;


// --- FUNCIONES MATEMÁTICAS ---

void hamming_window(float window[SIZE_WINDOW], int size) {
  for(int i=0; i < size; i++)
    window[i] = 0.53836f - 0.46164f * (cosf((2.0f * PI * i) / (size - 1)));
}

void preemphasis(float *input_signal) {

  const float coeff = 0.97f;

  for(int i = SHAPE_INPUT - 1; i >= 1; i--)
    input_signal[i] = input_signal[i] - input_signal[i-1] * coeff;
}

void windowing(float *input_signal, float **output_mat, float window[SIZE_WINDOW]) {

  int cont = 0;

  for(int i=0; i<NUMBER_OF_WINDOWS; i++) {

    for(int j=0; j<SIZE_WINDOW; j++) {

      output_mat[j][i] = window[j] * input_signal[cont];        

      cont++;
    }

    cont = cont - (SIZE_WINDOW - WINDOW_STEP);
  }
}

void fft_power_spectrum(float input_signal[SIZE_WINDOW], float output_signal[int(SIZE_WINDOW/2)+1]) {  

    for (int k = 0; k < SIZE_WINDOW; k++) {

        real_fft_plan->input[k] = input_signal[k];

    }

    fft_execute(real_fft_plan);

    output_signal[0] = fabsf(real_fft_plan->output[0]);

    for (int k = 1 ; k < real_fft_plan->size/2; k++) {

        float r = real_fft_plan->output[2*k];
        float im = real_fft_plan->output[2*k+1];

        output_signal[k] = (r*r + im*im); 
        //sqrtf(r*r + im*im);   ----------- cambio dado que la raíz cuadrada no reconocía patrones en la red neuronal,
        // se dejó el valor al cuadrado para mantener la relación de energía sin modificarla con la raíz.
        // Esto puede ayudar a la red a aprender patrones más fácilmente.
    }

    output_signal[real_fft_plan->size/2] = fabsf(real_fft_plan->output[1]);
}


// OPTIMIZACIÓN: Buffer interno para evitar tráfico en PSRAM

void spectrogram(float **input_matrix, float **output_matrix, int col, int row) {

  float vector_c[SIZE_WINDOW]; 
  float vector_fft[(SIZE_WINDOW/2)+1];

  int rows_fft = (SIZE_WINDOW/2)+1;

  for(int i=0; i<col; i++) {

    for(int j=0; j<row; j++) vector_c[j] = input_matrix[j][i];
    
    fft_power_spectrum(vector_c, vector_fft);
    
    for(int j=0; j<rows_fft; j++) output_matrix[j][i] = vector_fft[j];
  }
}


void triangular_filters(float **triangular_matrix) {

  float limits[NUM_BANDS_LIMITS];

  float frec_min_mel = 2595.0f * log10f(1.0f + (float)FREC_MIN / 700.0f);

  float frec_max_mel = 2595.0f * log10f(1.0f + (float)FREC_MAX / 700.0f);

  float separation_mel = (frec_max_mel - frec_min_mel) / (float)(NUM_BANDS_LIMITS - 1);

  limits[0] = frec_min_mel;

  for(int i=1; i<NUM_BANDS_LIMITS; i++) limits[i] = limits[i-1] + separation_mel;

  for(int i=0; i<NUM_BANDS_LIMITS; i++)
      limits[i] = 700.0f * (powf(10.0f, (limits[i] / 2595.0f)) - 1.0f);

  for(int i=0; i<NUM_BANDS_LIMITS; i++)
      limits[i] = floorf(((float)SIZE_WINDOW / 2.0f + 1.0f) * limits[i] / (float)FREC_MAX);


  for(int i=2; i<NUM_BANDS_LIMITS; i++) {

    for(int j=0; j < ((SIZE_WINDOW/2)+1); j++) {

      if(j >= limits[i-2] && j <= limits[i-1])

        triangular_matrix[i-2][j] = (j - limits[i-2]) / (limits[i-1] - limits[i-2]);

      else if(j > limits[i-1] && j <= limits[i])

        triangular_matrix[i-2][j] = (limits[i] - j) / (limits[i] - limits[i-1]);

      else

        triangular_matrix[i-2][j] = 0.0f;
    }
  }
}


// OPTIMIZACIÓN: Multiplicación matricial con buffer local

void mel_spectrogram(float **triangular_matrix, float **fft_matrix, float **mfcc_matrix) {

  int fft_rows = (SIZE_WINDOW / 2) + 1;

  for(int j = 0; j < NUMBER_OF_WINDOWS; j++) {

    float col_fft[fft_rows]; // Buffer en RAM interna

    for(int k = 0; k < fft_rows; k++) col_fft[k] = fft_matrix[k][j];

    for(int i = 0; i < MEL_BANDS; i++) {

      float suma_mel = 0.0f; 

      for(int k = 0; k < fft_rows; k++) {

        suma_mel += triangular_matrix[i][k] * col_fft[k];

      }

      // modificar a log   solo le puse el 10

      mfcc_matrix[i][j] = logf(suma_mel + 1e-6f);
      //13.0f * log10f(suma_mel + 1e-6f);  ---- cambio dado que multiplicacion por el 13 no reconocia patrones en la red neuronal
    }
  }
}


// --------------------------------------
// DCT optimizado (usa matriz precalculada)
// --------------------------------------

void dct1d(float vec_input[MEL_BANDS], float vec_output[MEL_BANDS]) {

  for (int k = 0; k < MEL_BANDS; k++) {    

    float sum = 0.0f;

    for (int n = 0; n < MEL_BANDS; n++)

        sum += vec_input[n] * dct_matrix[k][n];

    float f = (k == 0) ? sqrtf(1.0f / (4.0f * MEL_BANDS)) : sqrtf(1.0f / (2.0f * MEL_BANDS));

    vec_output[k] = 2.0f * sum * f;
  }
}


void dct_mat(float **mfcc_matrix) {

  float vector_in[MEL_BANDS];
  float vector_out[MEL_BANDS];

  for(int i = 0; i < NUMBER_OF_WINDOWS; i++) {  

    for(int j = 0; j < MEL_BANDS; j++) vector_in[j] = mfcc_matrix[j][i];

    dct1d(vector_in, vector_out);

    for(int j = 0; j < MEL_BANDS; j++) mfcc_matrix[j][i] = vector_out[j];
  }
}


// --------------------------------------
// INICIALIZACIÓN DE MATRICES
// --------------------------------------

void init_mfcc_matrices() {

    if (matrices_initialized) return;

    int fft_rows = (SIZE_WINDOW / 2) + 1;

    float *fft_in = (float*)heap_caps_malloc(SIZE_WINDOW * sizeof(float), MALLOC_CAP_INTERNAL);

    float *fft_out = (float*)heap_caps_malloc(SIZE_WINDOW * sizeof(float), MALLOC_CAP_INTERNAL);

    real_fft_plan = fft_init(SIZE_WINDOW, FFT_REAL, FFT_FORWARD, fft_in, fft_out);


    win_mat = (float**)heap_caps_malloc(SIZE_WINDOW * sizeof(float*), MALLOC_CAP_SPIRAM);

    for (int i = 0; i < SIZE_WINDOW; i++)
        win_mat[i] = (float*)heap_caps_malloc(NUMBER_OF_WINDOWS * sizeof(float), MALLOC_CAP_SPIRAM);


    fft_mat = (float**)heap_caps_malloc(fft_rows * sizeof(float*), MALLOC_CAP_SPIRAM);

    for (int i = 0; i < fft_rows; i++)
        fft_mat[i] = (float*)heap_caps_malloc(NUMBER_OF_WINDOWS * sizeof(float), MALLOC_CAP_SPIRAM);


    tri_mat = (float**)heap_caps_malloc(MEL_BANDS * sizeof(float*), MALLOC_CAP_SPIRAM);

    for (int i = 0; i < MEL_BANDS; i++) {

        tri_mat[i] = (float*)heap_caps_malloc(fft_rows * sizeof(float), MALLOC_CAP_SPIRAM);

        for(int j=0; j<fft_rows; j++) tri_mat[i][j] = 0.0f;
    }

    triangular_filters(tri_mat);

    // --------------------------------------
    // PRECALCULAR HAMMING
    // --------------------------------------

    if(!window_ready){
        hamming_window(window_global, SIZE_WINDOW);
        window_ready = true;
    }

    // --------------------------------------
    // PRECALCULAR MATRIZ DCT
    // --------------------------------------

    if(!dct_ready){

        for(int k=0;k<MEL_BANDS;k++){

            for(int n=0;n<MEL_BANDS;n++){

                dct_matrix[k][n] =
                    cosf((PI * k * (2.0f * n + 1.0f)) / (2.0f * MEL_BANDS));
            }
        }

        dct_ready = true;
    }

    matrices_initialized = true;

    Serial.println(">>> Matrices listas.");
}


void mfccs(float *input_signal, float **mfcc_matrix) {

    if (!matrices_initialized) init_mfcc_matrices();

    preemphasis(input_signal);

    windowing(input_signal, win_mat, window_global);

    spectrogram(win_mat, fft_mat, NUMBER_OF_WINDOWS, SIZE_WINDOW);

    mel_spectrogram(tri_mat, fft_mat, mfcc_matrix);

    dct_mat(mfcc_matrix);
}
