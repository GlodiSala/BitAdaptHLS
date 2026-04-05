// Delayed Scaling Testbench | S=4 NL=4
#include <stdio.h>
#include <math.h>
#include "transformer_top.h"
#include "hls_test_vectors.h"
#include "ap_int.h"

void embedding_layer(input_t input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void transformer_top(input_t input_flat[512], float output_flat[512]);
void attention_layer_0(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void rmsnorm_layer_0_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void ffn_block_0(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void rmsnorm_layer_0_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);void attention_layer_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void rmsnorm_layer_1_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void ffn_block_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void rmsnorm_layer_1_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);void attention_layer_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void rmsnorm_layer_2_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void ffn_block_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void rmsnorm_layer_2_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);void attention_layer_3(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void rmsnorm_layer_3_1(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void ffn_block_3(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]); void rmsnorm_layer_3_2(float input[SEQ_LEN][EMB_DIM], float output[SEQ_LEN][EMB_DIM]);
void output_layer(float input[SEQ_LEN][EMB_DIM], ap_int<5> x_int[SEQ_LEN][EMB_DIM], float s_act_out, float output_arr[SEQ_LEN][OUTPUT_DIM]);


static void check(const char* name, float* hls, const float* ref, int N, float tol) {
    float avg=0, mx=0; int f=0;
    float floor_v = 0.1f;
    for(int j=0;j<N;j++){
        float e=fabsf(hls[j]-ref[j]);
        float r=(fabsf(ref[j])>floor_v ? e/fabsf(ref[j]) : e/floor_v);
        avg+=r; if(r>mx)mx=r; if(r>tol)f++;
    }
    avg/=N;
    printf("%-24s avg_rel=%.4f max_rel=%.4f fail=%d/%d %s\n",
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
    printf("%-24s avg_abs=%.4f max_abs=%.4f fail=%d/%d %s\n",
           name,avg,mx,f,N,f==0?"ok":"WARN");
    printf("  HLS[0..3]: %+.5f %+.5f %+.5f %+.5f\n",hls[0],hls[1],hls[2],hls[3]);
    printf("  REF[0..3]: %+.5f %+.5f %+.5f %+.5f\n",ref[0],ref[1],ref[2],ref[3]);
}


int main() {
    input_t input_flat[512];
    float   output_flat[512];
    float   x[SEQ_LEN][EMB_DIM];
    float   buf[SEQ_LEN * EMB_DIM > SEQ_LEN * OUTPUT_DIM ?
                SEQ_LEN * EMB_DIM : SEQ_LEN * OUTPUT_DIM];

    for(int i=0;i<512;i++) input_flat[i]=(input_t)test_input_flat[i];

    printf("\n=== Delayed Scaling CSIM | S=%d D=%d NL=%d ===\n",
           SEQ_LEN, EMB_DIM, NUM_LAYERS);

    // Embedding
    {
        input_t in2d[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            in2d[si][j]=input_flat[si*EMB_DIM+j];
        embedding_layer(in2d, x);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j]=x[si][j];
        check("embedding", buf, ref_emb_flat, SEQ_LEN*EMB_DIM, 0.05f);
    }


    printf("\n--- Layer 0 ---\n");
    {
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            x[si][j] = ref_emb_flat[si*EMB_DIM+j];

        float attn_out[SEQ_LEN][EMB_DIM];
        attention_layer_0(x, attn_out);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = attn_out[si][j];
        check("L0_attn", buf, ref_attn_0_flat, SEQ_LEN*EMB_DIM, 0.10f);

        float norm1_in[SEQ_LEN][EMB_DIM], norm1_out[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm1_in[si][j] = x[si][j] + attn_out[si][j];
        rmsnorm_layer_0_1(norm1_in, norm1_out);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++) {
            x[si][j] = norm1_out[si][j];
            buf[si*EMB_DIM+j] = norm1_out[si][j];
        }
        check("L0_norm1", buf, ref_norm1_0_flat, SEQ_LEN*EMB_DIM, 0.10f);

        float ffn_out[SEQ_LEN][EMB_DIM];
        ffn_block_0(x, ffn_out);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = ffn_out[si][j];
        check_abs("L0_ffn", buf, ref_ffn_0_flat, SEQ_LEN*EMB_DIM, 1.0f);

        float norm2_in[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm2_in[si][j] = x[si][j] + ffn_out[si][j];
        rmsnorm_layer_0_2(norm2_in, x);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = x[si][j];
        check("L0_norm2", buf, ref_norm2_0_flat, SEQ_LEN*EMB_DIM, 0.10f);
    }
    printf("\n--- Layer 1 ---\n");
    {
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            x[si][j] = ref_norm2_0_flat[si*EMB_DIM+j];

        float attn_out[SEQ_LEN][EMB_DIM];
        attention_layer_1(x, attn_out);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = attn_out[si][j];
        check("L1_attn", buf, ref_attn_1_flat, SEQ_LEN*EMB_DIM, 0.10f);

        float norm1_in[SEQ_LEN][EMB_DIM], norm1_out[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm1_in[si][j] = x[si][j] + attn_out[si][j];
        rmsnorm_layer_1_1(norm1_in, norm1_out);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++) {
            x[si][j] = norm1_out[si][j];
            buf[si*EMB_DIM+j] = norm1_out[si][j];
        }
        check("L1_norm1", buf, ref_norm1_1_flat, SEQ_LEN*EMB_DIM, 0.10f);

        float ffn_out[SEQ_LEN][EMB_DIM];
        ffn_block_1(x, ffn_out);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = ffn_out[si][j];
        check_abs("L1_ffn", buf, ref_ffn_1_flat, SEQ_LEN*EMB_DIM, 1.0f);

        float norm2_in[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm2_in[si][j] = x[si][j] + ffn_out[si][j];
        rmsnorm_layer_1_2(norm2_in, x);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = x[si][j];
        check("L1_norm2", buf, ref_norm2_1_flat, SEQ_LEN*EMB_DIM, 0.10f);
    }
    printf("\n--- Layer 2 ---\n");
    {
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            x[si][j] = ref_norm2_1_flat[si*EMB_DIM+j];

        float attn_out[SEQ_LEN][EMB_DIM];
        attention_layer_2(x, attn_out);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = attn_out[si][j];
        check("L2_attn", buf, ref_attn_2_flat, SEQ_LEN*EMB_DIM, 0.10f);

        float norm1_in[SEQ_LEN][EMB_DIM], norm1_out[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm1_in[si][j] = x[si][j] + attn_out[si][j];
        rmsnorm_layer_2_1(norm1_in, norm1_out);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++) {
            x[si][j] = norm1_out[si][j];
            buf[si*EMB_DIM+j] = norm1_out[si][j];
        }
        check("L2_norm1", buf, ref_norm1_2_flat, SEQ_LEN*EMB_DIM, 0.10f);

        float ffn_out[SEQ_LEN][EMB_DIM];
        ffn_block_2(x, ffn_out);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = ffn_out[si][j];
        check_abs("L2_ffn", buf, ref_ffn_2_flat, SEQ_LEN*EMB_DIM, 1.0f);

        float norm2_in[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm2_in[si][j] = x[si][j] + ffn_out[si][j];
        rmsnorm_layer_2_2(norm2_in, x);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = x[si][j];
        check("L2_norm2", buf, ref_norm2_2_flat, SEQ_LEN*EMB_DIM, 0.10f);
    }
    printf("\n--- Layer 3 ---\n");
    {
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            x[si][j] = ref_norm2_2_flat[si*EMB_DIM+j];

        float attn_out[SEQ_LEN][EMB_DIM];
        attention_layer_3(x, attn_out);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = attn_out[si][j];
        check("L3_attn", buf, ref_attn_3_flat, SEQ_LEN*EMB_DIM, 0.10f);

        float norm1_in[SEQ_LEN][EMB_DIM], norm1_out[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm1_in[si][j] = x[si][j] + attn_out[si][j];
        rmsnorm_layer_3_1(norm1_in, norm1_out);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++) {
            x[si][j] = norm1_out[si][j];
            buf[si*EMB_DIM+j] = norm1_out[si][j];
        }
        check("L3_norm1", buf, ref_norm1_3_flat, SEQ_LEN*EMB_DIM, 0.10f);

        float ffn_out[SEQ_LEN][EMB_DIM];
        ffn_block_3(x, ffn_out);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = ffn_out[si][j];
        check_abs("L3_ffn", buf, ref_ffn_3_flat, SEQ_LEN*EMB_DIM, 1.0f);

        float norm2_in[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            norm2_in[si][j] = x[si][j] + ffn_out[si][j];
        rmsnorm_layer_3_2(norm2_in, x);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            buf[si*EMB_DIM+j] = x[si][j];
        check("L3_norm2", buf, ref_norm2_3_flat, SEQ_LEN*EMB_DIM, 0.10f);
    }

    // Output isolated
    printf("\n--- Output (from ref_norm2_3) ---\n");
    {
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++)
            x[si][j] = ref_norm2_3_flat[si*EMB_DIM+j];

        ap_int<5> x_int_out[SEQ_LEN][EMB_DIM];
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<EMB_DIM;j++) {
            float _v = roundf(x[si][j] * 9.12794719f);
            _v = _v < -15.0f ? -15.0f
               : (_v > 15.0f ? 15.0f : _v);
            x_int_out[si][j] = (ap_int<5>)_v;
        }
        float out2d[SEQ_LEN][OUTPUT_DIM];
        output_layer(x, x_int_out, 0.1095536575f, out2d);
        for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<OUTPUT_DIM;j++)
            buf[si*OUTPUT_DIM+j]=out2d[si][j];
        check_abs("output", buf, ref_output_flat, SEQ_LEN*OUTPUT_DIM, 0.5f);
    }

    // E2E
    printf("\n--- E2E transformer_top ---\n");
    transformer_top(input_flat, output_flat);

    for(int si=0;si<SEQ_LEN;si++) for(int j=0;j<OUTPUT_DIM;j++)
        buf[si*OUTPUT_DIM+j]=output_flat[si*OUTPUT_DIM+j];
    check_abs("E2E_output", buf, ref_output_flat, SEQ_LEN*OUTPUT_DIM, 0.5f);

    // Magnitude
    printf("\nE2E magnitude:\n");
    {
        float e2e_mag[256];
        for(int si=0;si<SEQ_LEN;si++) for(int k=0;k<OUTPUT_DIM/2;k++) {
            float re=output_flat[si*OUTPUT_DIM+k];
            float im=output_flat[si*OUTPUT_DIM+k+OUTPUT_DIM/2];
            e2e_mag[si*(OUTPUT_DIM/2)+k]=sqrtf(re*re+im*im);
        }
        check_abs("E2E_mag", e2e_mag, ref_mag_flat, 256, 0.5f);
        float tot=0; int ftot=0;
        for(int si=0;si<SEQ_LEN;si++) {
            float avg=0; int f=0;
            for(int k=0;k<OUTPUT_DIM/2;k++) {
                float e=fabsf(e2e_mag[si*(OUTPUT_DIM/2)+k]-ref_mag_flat[si*(OUTPUT_DIM/2)+k]);
                float r=ref_mag_flat[si*(OUTPUT_DIM/2)+k]>0.1f?e/ref_mag_flat[si*(OUTPUT_DIM/2)+k]:e/0.1f;
                avg+=r; if(r>0.10f)f++;
            }
            avg/=(OUTPUT_DIM/2); tot+=avg; ftot+=f;
            printf("  user[%d]: avg_rel=%.4f fail=%d/%d %s\n",si,avg,f,OUTPUT_DIM/2,f==0?"ok":"WARN");
        }
        printf("  E2E overall: avg_rel=%.4f total_fail=%d %s\n",
               tot/SEQ_LEN,ftot,ftot==0?"PASS":"FAIL");
    }
    return 0;
}
