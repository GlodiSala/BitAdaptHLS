// Delayed Scaling | Embedding | S=4 D=128
// w=ap_int<2> a=ap_int<4> acc=16b impl=fabric
// input_t=ap_fixed<16,2>
#include "transformer_top.h"
#include "embedding.h"
#include "ap_int.h"

void embedding_layer(input_t input[4][128], float output[4][128]) {

    const float s_act_emb = 0.0071089156f;
    ap_int<4> x_int[4][128];
    for (int si = 0; si < 4; si++) {
        Q_x_int: for (int j = 0; j < 128; j++) {
    #pragma HLS PIPELINE II=1
            float _v = roundf((float)input[si][j] * 140.66843055f);
            _v = _v < -15.0f ? -15.0f : (_v > 15.0f ? 15.0f : _v);
            x_int[si][j] = (ap_int<4>)_v;
        }
    }

    for (int s = 0; s < 4; s++) {
        for (int ii = 0; ii < 128; ii++) {
            ap_int<16> acc_int = 0;
            MAC_embedding: for (int j = 0; j < 128; j++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS BIND_OP variable=acc_int op=mul impl=fabric
                acc_int += (ap_int<16>)embedding_W[ii][j] * (ap_int<16>)x_int[s][j];
            }
            float acc_f = (float)acc_int * embedding_S[ii] * s_act_emb + (float)embedding_B[ii];
            output[s][ii] = acc_f;
        }
    }
}
