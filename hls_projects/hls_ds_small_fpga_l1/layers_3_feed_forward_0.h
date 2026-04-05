// Delayed Scaling | layers.3.feed_forward.0 | ap_int<3> x ap_int<3> -> impl=fabric
#pragma once
#include "ap_int.h"

typedef ap_int<3>    wi_layers_3_feed_forward_0_t;
typedef ap_int<3>    ai_layers_3_feed_forward_0_t;
typedef ap_int<16>     acc_layers_3_feed_forward_0_t;

#define LAYERS_3_FEED_FORWARD_0_OUT  512
#define LAYERS_3_FEED_FORWARD_0_IN   128
#define LAYERS_3_FEED_FORWARD_0_IMPL "fabric"

extern const wi_layers_3_feed_forward_0_t layers_3_feed_forward_0_W[512][128];
extern const float       layers_3_feed_forward_0_S[512];
extern const float       layers_3_feed_forward_0_B[512];