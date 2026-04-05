// Delayed Scaling Transformer Top | S=4 T=128 D=192 NL=4
#include "transformer_top.h"
#include "ap_int.h"
#include <math.h>

void embedding_layer(input_t input[SEQ_LEN][TOKEN_DIM], float output[SEQ_LEN][EMB_DIM]);
void attention_layer_0(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void rmsnorm_layer_0_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void ffn_block_0(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void rmsnorm_layer_0_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void attention_layer_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void rmsnorm_layer_1_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void ffn_block_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void rmsnorm_layer_1_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void attention_layer_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void rmsnorm_layer_2_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void ffn_block_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void rmsnorm_layer_2_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void attention_layer_3(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void rmsnorm_layer_3_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void ffn_block_3(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void rmsnorm_layer_3_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void output_layer(float input[SEQ_LEN][EMB_DIM], ap_int<6> x_int[SEQ_LEN][EMB_DIM], float s_act_out, float output_arr[SEQ_LEN][OUTPUT_DIM]);

void transformer_top(
    input_t input_flat[512],
    float   output_flat[512])
{
#pragma HLS INTERFACE ap_none port=input_flat
#pragma HLS INTERFACE ap_none port=output_flat
#pragma HLS INTERFACE ap_ctrl_none port=return

    // Reshape flat input to 2D token array [SEQ_LEN][TOKEN_DIM]
    input_t input_2d[SEQ_LEN][TOKEN_DIM];
    for (int si = 0; si < SEQ_LEN; si++)
        for (int j = 0; j < TOKEN_DIM; j++)
            input_2d[si][j] = input_flat[si * TOKEN_DIM + j];

    // All intermediate buffers in float
    float x[SEQ_LEN][EMB_DIM];

    // Embedding: TOKEN_DIM -> EMB_DIM
    embedding_layer(input_2d, x);


    // Layer 0
    {
        float attn_out[SEQ_LEN][EMB_DIM];
        attention_layer_0(x, attn_out);

        // Residual + RMSNorm1
        float norm1_in[SEQ_LEN][EMB_DIM];
        for (int si = 0; si < SEQ_LEN; si++)
            for (int j = 0; j < EMB_DIM; j++)
                norm1_in[si][j] = x[si][j] + attn_out[si][j];
        float norm1_out[SEQ_LEN][EMB_DIM];
        rmsnorm_layer_0_1(norm1_in, norm1_out);

        // FFN
        float ffn_out[SEQ_LEN][EMB_DIM];
        ffn_block_0(norm1_out, ffn_out);

        // Residual + RMSNorm2
        float norm2_in[SEQ_LEN][EMB_DIM];
        for (int si = 0; si < SEQ_LEN; si++)
            for (int j = 0; j < EMB_DIM; j++)
                norm2_in[si][j] = norm1_out[si][j] + ffn_out[si][j];
        rmsnorm_layer_0_2(norm2_in, x);
    }
    // Layer 1
    {
        float attn_out[SEQ_LEN][EMB_DIM];
        attention_layer_1(x, attn_out);

        // Residual + RMSNorm1
        float norm1_in[SEQ_LEN][EMB_DIM];
        for (int si = 0; si < SEQ_LEN; si++)
            for (int j = 0; j < EMB_DIM; j++)
                norm1_in[si][j] = x[si][j] + attn_out[si][j];
        float norm1_out[SEQ_LEN][EMB_DIM];
        rmsnorm_layer_1_1(norm1_in, norm1_out);

        // FFN
        float ffn_out[SEQ_LEN][EMB_DIM];
        ffn_block_1(norm1_out, ffn_out);

        // Residual + RMSNorm2
        float norm2_in[SEQ_LEN][EMB_DIM];
        for (int si = 0; si < SEQ_LEN; si++)
            for (int j = 0; j < EMB_DIM; j++)
                norm2_in[si][j] = norm1_out[si][j] + ffn_out[si][j];
        rmsnorm_layer_1_2(norm2_in, x);
    }
    // Layer 2
    {
        float attn_out[SEQ_LEN][EMB_DIM];
        attention_layer_2(x, attn_out);

        // Residual + RMSNorm1
        float norm1_in[SEQ_LEN][EMB_DIM];
        for (int si = 0; si < SEQ_LEN; si++)
            for (int j = 0; j < EMB_DIM; j++)
                norm1_in[si][j] = x[si][j] + attn_out[si][j];
        float norm1_out[SEQ_LEN][EMB_DIM];
        rmsnorm_layer_2_1(norm1_in, norm1_out);

        // FFN
        float ffn_out[SEQ_LEN][EMB_DIM];
        ffn_block_2(norm1_out, ffn_out);

        // Residual + RMSNorm2
        float norm2_in[SEQ_LEN][EMB_DIM];
        for (int si = 0; si < SEQ_LEN; si++)
            for (int j = 0; j < EMB_DIM; j++)
                norm2_in[si][j] = norm1_out[si][j] + ffn_out[si][j];
        rmsnorm_layer_2_2(norm2_in, x);
    }
    // Layer 3
    {
        float attn_out[SEQ_LEN][EMB_DIM];
        attention_layer_3(x, attn_out);

        // Residual + RMSNorm1
        float norm1_in[SEQ_LEN][EMB_DIM];
        for (int si = 0; si < SEQ_LEN; si++)
            for (int j = 0; j < EMB_DIM; j++)
                norm1_in[si][j] = x[si][j] + attn_out[si][j];
        float norm1_out[SEQ_LEN][EMB_DIM];
        rmsnorm_layer_3_1(norm1_in, norm1_out);

        // FFN
        float ffn_out[SEQ_LEN][EMB_DIM];
        ffn_block_3(norm1_out, ffn_out);

        // Residual + RMSNorm2
        float norm2_in[SEQ_LEN][EMB_DIM];
        for (int si = 0; si < SEQ_LEN; si++)
            for (int j = 0; j < EMB_DIM; j++)
                norm2_in[si][j] = norm1_out[si][j] + ffn_out[si][j];
        rmsnorm_layer_3_2(norm2_in, x);
    }

    // Output quantization to integer + Delayed Scaling MAC
    const float s_act_out = 0.0653444380f;
    ap_int<6> out_x_int[SEQ_LEN][EMB_DIM];
    for (int si = 0; si < SEQ_LEN; si++) {
        Q_OUT: for (int j = 0; j < EMB_DIM; j++) {
#pragma HLS PIPELINE II=1
            float _v = roundf((float)x[si][j] * 15.30352133f);
            _v = _v < -31.0f ? -31.0f : (_v > 31.0f ? 31.0f : _v);
            out_x_int[si][j] = (ap_int<6>)_v;
        }
    }

    float output_2d[SEQ_LEN][OUTPUT_DIM];
    output_layer(x, out_x_int, s_act_out, output_2d);

    for (int si = 0; si < SEQ_LEN; si++)
        for (int j = 0; j < OUTPUT_DIM; j++)
            output_flat[si * OUTPUT_DIM + j] = output_2d[si][j];
}
