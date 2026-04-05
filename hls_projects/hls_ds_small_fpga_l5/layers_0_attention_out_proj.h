// Delayed Scaling | layers.0.attention_out_proj | ap_int<2> x ap_int<2> -> impl=fabric
#pragma once
#include "ap_int.h"

typedef ap_int<2>    wi_layers_0_attention_out_proj_t;
typedef ap_int<2>    ai_layers_0_attention_out_proj_t;
typedef ap_int<16>     acc_layers_0_attention_out_proj_t;

#define LAYERS_0_ATTENTION_OUT_PROJ_OUT  128
#define LAYERS_0_ATTENTION_OUT_PROJ_IN   128
#define LAYERS_0_ATTENTION_OUT_PROJ_IMPL "fabric"

extern const wi_layers_0_attention_out_proj_t layers_0_attention_out_proj_W[128][128];
extern const float       layers_0_attention_out_proj_S[128];
extern const float       layers_0_attention_out_proj_B[128];