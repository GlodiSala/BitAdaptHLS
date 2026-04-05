// Delayed Scaling | layers.1.attention_in_proj | ap_int<2> x ap_int<2> -> impl=fabric
#pragma once
#include "ap_int.h"

typedef ap_int<2>    wi_layers_1_attention_in_proj_t;
typedef ap_int<2>    ai_layers_1_attention_in_proj_t;
typedef ap_int<16>     acc_layers_1_attention_in_proj_t;

#define LAYERS_1_ATTENTION_IN_PROJ_OUT  384
#define LAYERS_1_ATTENTION_IN_PROJ_IN   128
#define LAYERS_1_ATTENTION_IN_PROJ_IMPL "fabric"

extern const wi_layers_1_attention_in_proj_t layers_1_attention_in_proj_W[384][128];
extern const float       layers_1_attention_in_proj_S[384];
extern const float       layers_1_attention_in_proj_B[384];