// Delayed Scaling | Output layer | S=4 D=128 OUT=128
// w=ap_int<3> a=ap_int<5> acc=16b impl=fabric
#include "output.h"
#include "transformer_top.h"
#include "ap_int.h"

void output_layer(float input[4][128],
                  ap_int<5> x_int[4][128],
                  float s_act_out,
                  float output_arr[4][128]) {
    for (int s = 0; s < 4; s++) {
        for (int ii = 0; ii < 128; ii++) {
            ap_int<16> acc_int = 0;
            MAC_output: for (int j = 0; j < 128; j++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS BIND_OP variable=acc_int op=mul impl=fabric
                acc_int += (ap_int<16>)output_W[ii][j] * (ap_int<16>)x_int[s][j];
            }
            float acc_f = (float)acc_int * output_S[ii] * s_act_out + (float)output_B[ii];
            output_arr[s][ii] = acc_f;
        }
    }
}
