// RMSNorm layer 1 norm1 | S=4 D=128 | ms in float
#include "layers_1_norm1.h"
#include "transformer_top.h"
#include "hls_math.h"

void rmsnorm_layer_1_1(float input[4][128], float output[4][128]) {
    for (int s = 0; s < 4; s++) {
        float ms_f = 0.0f;
        RMS_SUM_1_1: for (int j = 0; j < 128; j++) {
#pragma HLS PIPELINE II=1
            float v = input[s][j];
            ms_f += v * v;
        }
        ms_f = ms_f / 128.0f + 1e-5f;
        float inv_rms = 1.0f / hls::sqrt(ms_f);
        RMS_SCALE_1_1: for (int j = 0; j < 128; j++) {
#pragma HLS PIPELINE II=1
            output[s][j] = input[s][j] * inv_rms * (float)layers_1_norm1_gamma[j];
        }
    }
}
