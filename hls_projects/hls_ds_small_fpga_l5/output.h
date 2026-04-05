// Delayed Scaling | output | ap_int<3> x ap_int<5> -> impl=fabric
#pragma once
#include "ap_int.h"

typedef ap_int<3>    wi_output_t;
typedef ap_int<5>    ai_output_t;
typedef ap_int<16>     acc_output_t;

#define OUTPUT_OUT  128
#define OUTPUT_IN   128
#define OUTPUT_IMPL "fabric"

extern const wi_output_t output_W[128][128];
extern const float       output_S[128];
extern const float       output_B[128];