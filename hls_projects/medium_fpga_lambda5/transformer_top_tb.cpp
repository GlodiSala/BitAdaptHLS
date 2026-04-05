// Delayed Scaling Testbench v4 | S=4 T=128 D=192 NL=4
// TEST 1: Blocs vraiment isolés — chaque bloc reçoit la ref numpy comme input
//         Norm reçoit résidu avec refs numpy, FFN reçoit ref_norm1 numpy
// TEST 2: Chaîne HLS complète — mesure accumulation réelle
// TEST 3: E2E transformer_top — validation finale avec cosine similarity
#include <stdio.h>
#include <math.h>
#include "transformer_top.h"
#include "hls_test_vectors.h"
#include "ap_int.h"

void embedding_layer(input_t input[SEQ_LEN][TOKEN_DIM], float output[SEQ_LEN][EMB_DIM]);
void transformer_top(input_t input_flat[512], float output_flat[512]);
void attention_layer_0(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void rmsnorm_layer_0_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void ffn_block_0(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void rmsnorm_layer_0_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void attention_layer_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void rmsnorm_layer_1_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void ffn_block_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void rmsnorm_layer_1_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void attention_layer_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void rmsnorm_layer_2_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void ffn_block_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void rmsnorm_layer_2_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void attention_layer_3(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void rmsnorm_layer_3_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void ffn_block_3(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void rmsnorm_layer_3_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void output_layer(float input[SEQ_LEN][EMB_DIM], ap_int<6> x_int[SEQ_LEN][EMB_DIM],
                  float s_act_out, float output_arr[SEQ_LEN][OUTPUT_DIM]);


static void check(const char* name, float* hls, const float* ref, int N, float tol) {
    float avg=0, mx=0; int f=0; float floor_v=0.05f;
    for(int j=0;j<N;j++){
        float e=fabsf(hls[j]-ref[j]);
        float r=(fabsf(ref[j])>floor_v?e/fabsf(ref[j]):e/floor_v);
        avg+=r; if(r>mx)mx=r; if(r>tol)f++;
    }
    avg/=N;
    printf("%-28s avg_rel=%.4f max_rel=%.4f fail=%d/%d %s\n",
           name,avg,mx,f,N,f==0?"ok":"WARN");
    printf("  HLS[0..3]: %+.5f %+.5f %+.5f %+.5f\n",hls[0],hls[1],hls[2],hls[3]);
    printf("  REF[0..3]: %+.5f %+.5f %+.5f %+.5f\n",ref[0],ref[1],ref[2],ref[3]);
}
static void check_abs(const char* name, float* hls, const float* ref, int N, float tol) {
    float avg=0, mx=0; int f=0;
    for(int j=0;j<N;j++){
        float e=fabsf(hls[j]-ref[j]);
        avg+=e; if(e>mx)mx=e; if(e>tol)f++;
    }
    avg/=N;
    printf("%-28s avg_abs=%.4f max_abs=%.4f fail=%d/%d %s\n",
           name,avg,mx,f,N,f==0?"ok":"WARN");
    printf("  HLS[0..3]: %+.5f %+.5f %+.5f %+.5f\n",hls[0],hls[1],hls[2],hls[3]);
    printf("  REF[0..3]: %+.5f %+.5f %+.5f %+.5f\n",ref[0],ref[1],ref[2],ref[3]);
}
static float cosine_sim(const float* a, const float* b, int N) {
    float dot=0, na=0, nb=0;
    for(int j=0;j<N;j++){dot+=a[j]*b[j]; na+=a[j]*a[j]; nb+=b[j]*b[j];}
    return dot/(sqrtf(na)*sqrtf(nb)+1e-9f);
}


static void run_layer_0(float x[SEQ_LEN][EMB_DIM]) {
    float attn_out[SEQ_LEN][EMB_DIM];
    attention_layer_0(x, attn_out);
    float norm1_in[SEQ_LEN][EMB_DIM], norm1_out[SEQ_LEN][EMB_DIM];
    for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
        norm1_in[si][j] = x[si][j] + attn_out[si][j];
    rmsnorm_layer_0_1(norm1_in, norm1_out);
    float ffn_out[SEQ_LEN][EMB_DIM];
    ffn_block_0(norm1_out, ffn_out);
    float norm2_in[SEQ_LEN][EMB_DIM];
    for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
        norm2_in[si][j] = norm1_out[si][j] + ffn_out[si][j];
    rmsnorm_layer_0_2(norm2_in, x);
}
static void run_layer_1(float x[SEQ_LEN][EMB_DIM]) {
    float attn_out[SEQ_LEN][EMB_DIM];
    attention_layer_1(x, attn_out);
    float norm1_in[SEQ_LEN][EMB_DIM], norm1_out[SEQ_LEN][EMB_DIM];
    for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
        norm1_in[si][j] = x[si][j] + attn_out[si][j];
    rmsnorm_layer_1_1(norm1_in, norm1_out);
    float ffn_out[SEQ_LEN][EMB_DIM];
    ffn_block_1(norm1_out, ffn_out);
    float norm2_in[SEQ_LEN][EMB_DIM];
    for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
        norm2_in[si][j] = norm1_out[si][j] + ffn_out[si][j];
    rmsnorm_layer_1_2(norm2_in, x);
}
static void run_layer_2(float x[SEQ_LEN][EMB_DIM]) {
    float attn_out[SEQ_LEN][EMB_DIM];
    attention_layer_2(x, attn_out);
    float norm1_in[SEQ_LEN][EMB_DIM], norm1_out[SEQ_LEN][EMB_DIM];
    for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
        norm1_in[si][j] = x[si][j] + attn_out[si][j];
    rmsnorm_layer_2_1(norm1_in, norm1_out);
    float ffn_out[SEQ_LEN][EMB_DIM];
    ffn_block_2(norm1_out, ffn_out);
    float norm2_in[SEQ_LEN][EMB_DIM];
    for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
        norm2_in[si][j] = norm1_out[si][j] + ffn_out[si][j];
    rmsnorm_layer_2_2(norm2_in, x);
}
static void run_layer_3(float x[SEQ_LEN][EMB_DIM]) {
    float attn_out[SEQ_LEN][EMB_DIM];
    attention_layer_3(x, attn_out);
    float norm1_in[SEQ_LEN][EMB_DIM], norm1_out[SEQ_LEN][EMB_DIM];
    for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
        norm1_in[si][j] = x[si][j] + attn_out[si][j];
    rmsnorm_layer_3_1(norm1_in, norm1_out);
    float ffn_out[SEQ_LEN][EMB_DIM];
    ffn_block_3(norm1_out, ffn_out);
    float norm2_in[SEQ_LEN][EMB_DIM];
    for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
        norm2_in[si][j] = norm1_out[si][j] + ffn_out[si][j];
    rmsnorm_layer_3_2(norm2_in, x);
}

static void apply_output(float x[SEQ_LEN][EMB_DIM], float out[SEQ_LEN][OUTPUT_DIM]) {
    ap_int<6> xi[SEQ_LEN][EMB_DIM];
    for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++) {
        float _v = roundf(x[si][j] * 15.30352133f);
        _v = _v < -31.0f ? -31.0f
           : (_v > 31.0f ? 31.0f : _v);
        xi[si][j] = (ap_int<6>)_v;
    }
    output_layer(x, xi, 0.0653444380f, out);
}

int main() {
    input_t input_flat[512];
    float   output_flat[512];
    float   x[SEQ_LEN][EMB_DIM];
    float   buf[SEQ_LEN*EMB_DIM > SEQ_LEN*OUTPUT_DIM ?
                SEQ_LEN*EMB_DIM : SEQ_LEN*OUTPUT_DIM];

    for(int i=0;i<512;i++) input_flat[i]=(input_t)test_input_flat[i];

    printf("\n=== Delayed Scaling CSIM v4 | S=%d T=%d D=%d NL=%d ===\n",
           SEQ_LEN, TOKEN_DIM, EMB_DIM, NUM_LAYERS);

    //  TEST 1: Blocs vraiment isolés 
    printf("\n=== TEST 1: Blocs isolés (ref numpy comme input) ===\n");
    {
        // Embedding
        input_t in2d[SEQ_LEN][TOKEN_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<TOKEN_DIM;j++)
            in2d[si][j] = input_flat[si*TOKEN_DIM+j];
        embedding_layer(in2d, x);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = x[si][j];
        check("embedding", buf, ref_emb_flat, SEQ_LEN*EMB_DIM, 0.15f);
    }

    //  Layer 0 vraiment isolé 
    {
        // Attn: reçoit ref_in directement
        float ref_in_0[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            ref_in_0[si][j] = ref_emb_flat[si*EMB_DIM+j];
        float attn_out_0[SEQ_LEN][EMB_DIM];
        attention_layer_0(ref_in_0, attn_out_0);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = attn_out_0[si][j];
        check("L0_attn", buf, ref_attn_0_flat, SEQ_LEN*EMB_DIM, 0.15f);

        // Norm1: résidu avec refs numpy (attn HLS + ref_in numpy)
        float norm1_in_0[SEQ_LEN][EMB_DIM], norm1_out_0[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm1_in_0[si][j] = ref_emb_flat[si*EMB_DIM+j] + ref_attn_0_flat[si*EMB_DIM+j];
        rmsnorm_layer_0_1(norm1_in_0, norm1_out_0);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = norm1_out_0[si][j];
        check("L0_norm1", buf, ref_norm1_0_flat, SEQ_LEN*EMB_DIM, 0.15f);

        // FFN: reçoit ref_norm1 numpy directement (pas HLS norm1)
        float ffn_in_0[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            ffn_in_0[si][j] = ref_norm1_0_flat[si*EMB_DIM+j];
        float ffn_out_0[SEQ_LEN][EMB_DIM];
        ffn_block_0(ffn_in_0, ffn_out_0);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = ffn_out_0[si][j];
        check_abs("L0_ffn", buf, ref_ffn_0_flat, SEQ_LEN*EMB_DIM, 2.0f);

        // Norm2: résidu avec refs numpy (ffn HLS + ref_norm1 numpy)
        float norm2_in_0[SEQ_LEN][EMB_DIM], norm2_out_0[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm2_in_0[si][j] = ref_norm1_0_flat[si*EMB_DIM+j]
                                 + ref_ffn_0_flat[si*EMB_DIM+j];
        rmsnorm_layer_0_2(norm2_in_0, norm2_out_0);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = norm2_out_0[si][j];
        check("L0_norm2", buf, ref_norm2_0_flat, SEQ_LEN*EMB_DIM, 0.15f);
    }
    //  Layer 1 vraiment isolé 
    {
        // Attn: reçoit ref_in directement
        float ref_in_1[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            ref_in_1[si][j] = ref_norm2_0_flat[si*EMB_DIM+j];
        float attn_out_1[SEQ_LEN][EMB_DIM];
        attention_layer_1(ref_in_1, attn_out_1);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = attn_out_1[si][j];
        check("L1_attn", buf, ref_attn_1_flat, SEQ_LEN*EMB_DIM, 0.15f);

        // Norm1: résidu avec refs numpy (attn HLS + ref_in numpy)
        float norm1_in_1[SEQ_LEN][EMB_DIM], norm1_out_1[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm1_in_1[si][j] = ref_norm2_0_flat[si*EMB_DIM+j] + ref_attn_1_flat[si*EMB_DIM+j];
        rmsnorm_layer_1_1(norm1_in_1, norm1_out_1);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = norm1_out_1[si][j];
        check("L1_norm1", buf, ref_norm1_1_flat, SEQ_LEN*EMB_DIM, 0.15f);

        // FFN: reçoit ref_norm1 numpy directement (pas HLS norm1)
        float ffn_in_1[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            ffn_in_1[si][j] = ref_norm1_1_flat[si*EMB_DIM+j];
        float ffn_out_1[SEQ_LEN][EMB_DIM];
        ffn_block_1(ffn_in_1, ffn_out_1);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = ffn_out_1[si][j];
        check_abs("L1_ffn", buf, ref_ffn_1_flat, SEQ_LEN*EMB_DIM, 2.0f);

        // Norm2: résidu avec refs numpy (ffn HLS + ref_norm1 numpy)
        float norm2_in_1[SEQ_LEN][EMB_DIM], norm2_out_1[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm2_in_1[si][j] = ref_norm1_1_flat[si*EMB_DIM+j]
                                 + ref_ffn_1_flat[si*EMB_DIM+j];
        rmsnorm_layer_1_2(norm2_in_1, norm2_out_1);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = norm2_out_1[si][j];
        check("L1_norm2", buf, ref_norm2_1_flat, SEQ_LEN*EMB_DIM, 0.15f);
    }
    //  Layer 2 vraiment isolé 
    {
        // Attn: reçoit ref_in directement
        float ref_in_2[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            ref_in_2[si][j] = ref_norm2_1_flat[si*EMB_DIM+j];
        float attn_out_2[SEQ_LEN][EMB_DIM];
        attention_layer_2(ref_in_2, attn_out_2);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = attn_out_2[si][j];
        check("L2_attn", buf, ref_attn_2_flat, SEQ_LEN*EMB_DIM, 0.15f);

        // Norm1: résidu avec refs numpy (attn HLS + ref_in numpy)
        float norm1_in_2[SEQ_LEN][EMB_DIM], norm1_out_2[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm1_in_2[si][j] = ref_norm2_1_flat[si*EMB_DIM+j] + ref_attn_2_flat[si*EMB_DIM+j];
        rmsnorm_layer_2_1(norm1_in_2, norm1_out_2);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = norm1_out_2[si][j];
        check("L2_norm1", buf, ref_norm1_2_flat, SEQ_LEN*EMB_DIM, 0.15f);

        // FFN: reçoit ref_norm1 numpy directement (pas HLS norm1)
        float ffn_in_2[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            ffn_in_2[si][j] = ref_norm1_2_flat[si*EMB_DIM+j];
        float ffn_out_2[SEQ_LEN][EMB_DIM];
        ffn_block_2(ffn_in_2, ffn_out_2);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = ffn_out_2[si][j];
        check_abs("L2_ffn", buf, ref_ffn_2_flat, SEQ_LEN*EMB_DIM, 2.0f);

        // Norm2: résidu avec refs numpy (ffn HLS + ref_norm1 numpy)
        float norm2_in_2[SEQ_LEN][EMB_DIM], norm2_out_2[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm2_in_2[si][j] = ref_norm1_2_flat[si*EMB_DIM+j]
                                 + ref_ffn_2_flat[si*EMB_DIM+j];
        rmsnorm_layer_2_2(norm2_in_2, norm2_out_2);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = norm2_out_2[si][j];
        check("L2_norm2", buf, ref_norm2_2_flat, SEQ_LEN*EMB_DIM, 0.15f);
    }
    //  Layer 3 vraiment isolé 
    {
        // Attn: reçoit ref_in directement
        float ref_in_3[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            ref_in_3[si][j] = ref_norm2_2_flat[si*EMB_DIM+j];
        float attn_out_3[SEQ_LEN][EMB_DIM];
        attention_layer_3(ref_in_3, attn_out_3);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = attn_out_3[si][j];
        check("L3_attn", buf, ref_attn_3_flat, SEQ_LEN*EMB_DIM, 0.15f);

        // Norm1: résidu avec refs numpy (attn HLS + ref_in numpy)
        float norm1_in_3[SEQ_LEN][EMB_DIM], norm1_out_3[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm1_in_3[si][j] = ref_norm2_2_flat[si*EMB_DIM+j] + ref_attn_3_flat[si*EMB_DIM+j];
        rmsnorm_layer_3_1(norm1_in_3, norm1_out_3);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = norm1_out_3[si][j];
        check("L3_norm1", buf, ref_norm1_3_flat, SEQ_LEN*EMB_DIM, 0.15f);

        // FFN: reçoit ref_norm1 numpy directement (pas HLS norm1)
        float ffn_in_3[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            ffn_in_3[si][j] = ref_norm1_3_flat[si*EMB_DIM+j];
        float ffn_out_3[SEQ_LEN][EMB_DIM];
        ffn_block_3(ffn_in_3, ffn_out_3);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = ffn_out_3[si][j];
        check_abs("L3_ffn", buf, ref_ffn_3_flat, SEQ_LEN*EMB_DIM, 2.0f);

        // Norm2: résidu avec refs numpy (ffn HLS + ref_norm1 numpy)
        float norm2_in_3[SEQ_LEN][EMB_DIM], norm2_out_3[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm2_in_3[si][j] = ref_norm1_3_flat[si*EMB_DIM+j]
                                 + ref_ffn_3_flat[si*EMB_DIM+j];
        rmsnorm_layer_3_2(norm2_in_3, norm2_out_3);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = norm2_out_3[si][j];
        check("L3_norm2", buf, ref_norm2_3_flat, SEQ_LEN*EMB_DIM, 0.15f);
    }
    // Output isolé
    {
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            x[si][j] = ref_norm2_3_flat[si*EMB_DIM+j];
        float out2d[SEQ_LEN][OUTPUT_DIM];
        apply_output(x, out2d);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<OUTPUT_DIM;j++)
            buf[si*OUTPUT_DIM+j] = out2d[si][j];
        check_abs("output_isolé", buf, ref_output_flat, SEQ_LEN*OUTPUT_DIM, 0.5f);
    }

    //  TEST 2: Chaîne HLS 
    printf("\n=== TEST 2: Chaîne HLS (accumulation) ===\n");
    {
        float xc[SEQ_LEN][EMB_DIM];
        input_t in2d[SEQ_LEN][TOKEN_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<TOKEN_DIM;j++)
            in2d[si][j] = input_flat[si*TOKEN_DIM+j];
        embedding_layer(in2d, xc);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = xc[si][j];
        check("chain_emb", buf, ref_emb_flat, SEQ_LEN*EMB_DIM, 0.15f);

        run_layer_0(xc);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = xc[si][j];
        check("chain_L0", buf, ref_norm2_0_flat, SEQ_LEN*EMB_DIM, 0.50f);
        run_layer_1(xc);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = xc[si][j];
        check("chain_L1", buf, ref_norm2_1_flat, SEQ_LEN*EMB_DIM, 0.50f);
        run_layer_2(xc);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = xc[si][j];
        check("chain_L2", buf, ref_norm2_2_flat, SEQ_LEN*EMB_DIM, 0.50f);
        run_layer_3(xc);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = xc[si][j];
        check("chain_L3", buf, ref_norm2_3_flat, SEQ_LEN*EMB_DIM, 0.50f);
        float out2d[SEQ_LEN][OUTPUT_DIM];
        apply_output(xc, out2d);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<OUTPUT_DIM;j++)
            buf[si*OUTPUT_DIM+j] = out2d[si][j];
        check_abs("chain_output", buf, ref_output_flat, SEQ_LEN*OUTPUT_DIM, 2.0f);
    }

    //  TEST 3: E2E transformer_top 
    printf("\n=== TEST 3: E2E transformer_top ===\n");
    transformer_top(input_flat, output_flat);

    // Cosine similarity par user — ce qui compte pour le precoding
    printf("  Cosine similarity HLS vs ref (direction du beamforming):\n");
    int all_ok = 1;
    for(int si=0;si<SEQ_LEN;si++) {
        float cos = cosine_sim(
            output_flat + si*OUTPUT_DIM,
            ref_output_flat + si*OUTPUT_DIM,
            OUTPUT_DIM);
        int ok = (cos > 0.80f);
        if(!ok) all_ok=0;
        printf("    user[%d]: cos_sim=%+.4f %s\n", si, cos, ok?"OK":"WARN");
    }
    printf("  E2E: %s\n", all_ok?"PASS":"FAIL");

    // Magnitude E2E
    printf("  Magnitude HLS vs ref:\n");
    for(int si=0;si<SEQ_LEN;si++) {
        float mag_hls=0, mag_ref=0;
        for(int k=0;k<OUTPUT_DIM/2;k++) {
            float rh=output_flat[si*OUTPUT_DIM+k];
            float ih=output_flat[si*OUTPUT_DIM+k+OUTPUT_DIM/2];
            float rr=ref_output_flat[si*OUTPUT_DIM+k];
            float ir=ref_output_flat[si*OUTPUT_DIM+k+OUTPUT_DIM/2];
            mag_hls += rh*rh+ih*ih;
            mag_ref += rr*rr+ir*ir;
        }
        printf("    user[%d]: |HLS|=%.3f |REF|=%.3f ratio=%.3f\n",
               si, sqrtf(mag_hls), sqrtf(mag_ref),
               sqrtf(mag_hls)/(sqrtf(mag_ref)+1e-9f));
    }
    return 0;
}
