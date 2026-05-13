#ifndef MFCC_H
#define MFCC_H

//Declaraciones
//*********************************************************************************

//Variables para FFT
//------------------------------------------------------------------------------------------
#define SAMPLING_RATE 16000//4096 cambio a 16000 para micrófono I2S
#define TOTAL_TIME 2
#define SHAPE_INPUT SAMPLING_RATE*TOTAL_TIME //Tamaño de la señal de entrada

//Variables para Ventaneo
//------------------------------------------------------------------------------------------
#define SIZE_WINDOW 512 //antes era 256    ahora es diferente para que coincida con el entrenamiento 16khz
#define WINDOW_STEP 256 // antes 128
#define NUMBER_OF_WINDOWS (((SHAPE_INPUT)-(SIZE_WINDOW))/(WINDOW_STEP) + 1)   
#define SHAPE_FFT (SHAPE_INPUT/2)

//Variables para MFCC
//------------------------------------------------------------------------------------------
#define FREC_MIN 20
#define FREC_MAX (SAMPLING_RATE/2)
#define MEL_BANDS 20
#define NUM_BANDS_LIMITS (MEL_BANDS+2)

//Prototipos
//------------------------------------------------------------------------------------------
void hamming_window(float window[SIZE_WINDOW], int size);
void preemphasis(float *input_signal);
void fft_power_spectrum(float input_signal[SIZE_WINDOW], float output_signal[int(SIZE_WINDOW/2)+1]);
void windowing(float *input_signal, float **output_mat, float window[SIZE_WINDOW]);
void spectrogram(float **input_matrix, float **output_matrix, int col, int row);
void triangular_filters(float **triangular_matrix);
void mel_spectrogram(float **triangular_matrix, float **fft_matrix, float **mfcc_matrix);
void dct1d(float vec_input[MEL_BANDS], float vec_output[MEL_BANDS]);
void dct_mat(float **mfcc_matrix);
void mfccs(float *input_signal, float **mfcc_matrix);

#endif