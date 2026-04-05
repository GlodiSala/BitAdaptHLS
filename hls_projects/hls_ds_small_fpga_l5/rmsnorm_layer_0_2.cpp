// RMSNorm layer 0 norm2 | S=4 D=128 | ms in float
#include "layers_0_norm2.h"
#include "transformer_top.h"
#include "hls_math.h"

void rmsnorm_layer_0_2(float input[4][128], float output[4][128]) {
    for (int s = 0; s < 4; s++) {
        float ms_f = 0.0f;
        RMS_SUM_0_2: for (int j = 0; j < 128; j++) {
#pragma HLS PIPELINE II=1
            float v = input[s][j];
            ms_f += v * v;
        }
        ms_f = ms_f / 128.0f + 1e-5f;
        float inv_rms = 1.0f / hls::sqrt(ms_f);
        RMS_SCALE_0_2: for (int j = 0; j < 128; j++) {
#pragma HLS PIPELINE II=1
            output[s][j] = input[s][j] * inv_rms * (float)layers_0_norm2_gamma[j];
        }
    }
}
