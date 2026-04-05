// Delayed Scaling | layers.2.attention_out_proj | ap_int<3> x ap_int<4> -> impl=fabric
#pragma once
#include "ap_int.h"

typedef ap_int<3>    wi_layers_2_attention_out_proj_t;
typedef ap_int<4>    ai_layers_2_attention_out_proj_t;
typedef ap_int<16>     acc_layers_2_attention_out_proj_t;

#define LAYERS_2_ATTENTION_OUT_PROJ_OUT  128
#define LAYERS_2_ATTENTION_OUT_PROJ_IN   128
#define LAYERS_2_ATTENTION_OUT_PROJ_IMPL "fabric"

extern const wi_layers_2_attention_out_proj_t layers_2_attention_out_proj_W[128][128];
extern const float       layers_2_attention_out_proj_S[128];
extern const float       layers_2_attention_out_proj_B[128];