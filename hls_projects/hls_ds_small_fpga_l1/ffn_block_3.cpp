// Delayed Scaling | FFN layer 3 | S=4 D=128 H=512
// ff0: w=ap_int<3> a=ap_int<3> impl=fabric
// ff1: w=ap_int<3> a=ap_int<5> impl=fabric
#include "layers_3_feed_forward_0.h"
#include "layers_3_feed_forward_3.h"
#include "transformer_top.h"
#include "ap_int.h"
#include <math.h>

void ffn_block_3(float input[4][128], float output[4][128]) {

    const float s_act0 = 0.3832995296f;
    ap_int<3> x_int0[4][128];
    for (int si = 0; si < 4; si++) {
        Q_x_int0: for (int j = 0; j < 128; j++) {
    #pragma HLS PIPELINE II=1
            float _v = roundf((float)input[si][j] * 2.60892572f);
            _v = _v < -7.0f ? -7.0f : (_v > 7.0f ? 7.0f : _v);
            x_int0[si][j] = (ap_int<3>)_v;
        }
    }

    float mid[4][512];
    for (int s = 0; s < 4; s++) {
        for (int ii = 0; ii < 512; ii++) {
            ap_int<16> acc_int = 0;
            MAC_layers_3_feed_forward_0: for (int j = 0; j < 128; j++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS BIND_OP variable=acc_int op=mul impl=fabric
                acc_int += (ap_int<16>)layers_3_feed_forward_0_W[ii][j] * (ap_int<16>)x_int0[s][j];
            }
            float acc_f = (float)acc_int * layers_3_feed_forward_0_S[ii] * s_act0 + (float)layers_3_feed_forward_0_B[ii];
            acc_f = acc_f > 0.0f ? acc_f : 0.0f;
            mid[s][ii] = acc_f;
        }
    }

    const float s_act1 = 0.4409630895f;
    ap_int<5> x_int1[4][512];
    for (int si = 0; si < 4; si++) {
        Q_x_int1: for (int j = 0; j < 512; j++) {
    #pragma HLS PIPELINE II=1
            float _v = roundf((float)mid[si][j] * 2.26776350f);
            _v = _v < -15.0f ? -15.0f : (_v > 15.0f ? 15.0f : _v);
            x_int1[si][j] = (ap_int<5>)_v;
        }
    }

    for (int s = 0; s < 4; s++) {
        for (int ii = 0; ii < 128; ii++) {
            ap_int<18> acc_int = 0;
            MAC_layers_3_feed_forward_3: for (int j = 0; j < 512; j++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS BIND_OP variable=acc_int op=mul impl=fabric
                acc_int += (ap_int<18>)layers_3_feed_forward_3_W[ii][j] * (ap_int<18>)x_int1[s][j];
            }
            float acc_f = (float)acc_int * layers_3_feed_forward_3_S[ii] * s_act1 + (float)layers_3_feed_forward_3_B[ii];
            output[s][ii] = acc_f;
        }
    }
}
