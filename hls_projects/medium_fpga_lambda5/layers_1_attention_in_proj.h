// Delayed Scaling | layers.1.attention_in_proj | ap_int<3> x ap_int<3> -> impl=fabric
// LSQ thd=+-3 (b_w=2), HLS type=ap_int<3>
#pragma once
#include "ap_int.h"

typedef ap_int<3>    wi_layers_1_attention_in_proj_t;
typedef ap_int<3>    ai_layers_1_attention_in_proj_t;
typedef ap_int<16>        acc_layers_1_attention_in_proj_t;

#define LAYERS_1_ATTENTION_IN_PROJ_OUT  576
#define LAYERS_1_ATTENTION_IN_PROJ_IN   192
#define LAYERS_1_ATTENTION_IN_PROJ_IMPL "fabric"

extern const wi_layers_1_attention_in_proj_t layers_1_attention_in_proj_W[576][192];
extern const float        layers_1_attention_in_proj_S[576];
extern const float        layers_1_attention_in_proj_B[576];