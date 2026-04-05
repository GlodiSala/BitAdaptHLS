// Delayed Scaling | output | ap_int<4> x ap_int<6> -> impl=fabric
// LSQ thd=+-7 (b_w=3), HLS type=ap_int<4>
#pragma once
#include "ap_int.h"

typedef ap_int<4>    wi_output_t;
typedef ap_int<6>    ai_output_t;
typedef ap_int<18>        acc_output_t;

#define OUTPUT_OUT  128
#define OUTPUT_IN   192
#define OUTPUT_IMPL "fabric"

extern const wi_output_t output_W[128][192];
extern const float        output_S[128];
extern const float        output_B[128];