// Delayed Scaling | Attention layer 3 | S=4 H=4 HD=32
// QKV: w=ap_int<3> a=ap_int<4> impl=fabric
// OutProj: float MAC (ctx is post-softmax float)
#include "layers_3_attention_in_proj.h"
#include "layers_3_attention_out_proj.h"
#include "transformer_top.h"
#include "ap_int.h"
#include <math.h>

void attention_layer_3(float input[4][128], float output[4][128]) {

    const float s_act_attn = 0.1013758779f;
    ap_int<4> x_int[4][128];
    for (int si = 0; si < 4; si++) {
        Q_x_int: for (int j = 0; j < 128; j++) {
    #pragma HLS PIPELINE II=1
            float _v = roundf((float)input[si][j] * 9.86427956f);
            _v = _v < -15.0f ? -15.0f : (_v > 15.0f ? 15.0f : _v);
            x_int[si][j] = (ap_int<4>)_v;
        }
    }


    float qkv[4][384];
    for (int s = 0; s < 4; s++) {
        for (int ii = 0; ii < 384; ii++) {
            ap_int<16> acc_int = 0;
            MAC_layers_3_attention_in_proj: for (int j = 0; j < 128; j++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS BIND_OP variable=acc_int op=mul impl=fabric
                acc_int += (ap_int<16>)layers_3_attention_in_proj_W[ii][j] * (ap_int<16>)x_int[s][j];
            }
            float acc_f = (float)acc_int * layers_3_attention_in_proj_S[ii] * s_act_attn + (float)layers_3_attention_in_proj_B[ii];
            qkv[s][ii] = acc_f;
        }
    }


    // Multi-head attention scores + softmax (float — required for numerical stability)
    float ctx[4][128];
    for (int h = 0; h < 4; h++) {
        int qoff = h * 32;
        int koff = 128 + h * 32;
        int voff = 256 + h * 32;

        float scores[4][4];
        for (int qi = 0; qi < 4; qi++) {
            SCORE_3_h: for (int kj = 0; kj < 4; kj++) {
#pragma HLS PIPELINE II=1
                float dot = 0.0f;
                for (int d = 0; d < 32; d++)
                    dot += qkv[qi][qoff+d] * qkv[kj][koff+d];
                scores[qi][kj] = dot * 0.17677670f;
            }
        }

        for (int qi = 0; qi < 4; qi++) {
            float mx = scores[qi][0];
            for (int kj = 1; kj < 4; kj++)
                if (scores[qi][kj] > mx) mx = scores[qi][kj];
            float aw[4], sum = 0.0f;
            SOFTMAX_3: for (int kj = 0; kj < 4; kj++) {
#pragma HLS PIPELINE II=1
                aw[kj] = expf(scores[qi][kj] - mx);
                sum += aw[kj];
            }
            float inv_sum = 1.0f / (sum + 1e-9f);
            CTX_3: for (int d = 0; d < 32; d++) {
#pragma HLS PIPELINE II=1
                float c = 0.0f;
                for (int kj = 0; kj < 4; kj++)
                    c += aw[kj] * inv_sum * qkv[kj][voff+d];
                ctx[qi][h*32+d] = c;
            }
        }
    }

    // Out projection — ctx is float (post-softmax), use float MAC
    for (int s = 0; s < 4; s++) {
        for (int r = 0; r < 128; r++) {
            float acc = (float)layers_3_attention_out_proj_B[r];
            OUT_PROJ_3: for (int c = 0; c < 128; c++) {
#pragma HLS PIPELINE II=1
                // W stored as int, dequantize on the fly
                acc += (float)layers_3_attention_out_proj_W[r][c] * layers_3_attention_out_proj_S[r] * ctx[s][c];
            }
            output[s][r] = acc;
        }
    }
}
