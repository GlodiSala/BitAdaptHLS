// Delayed Scaling | layers.2.feed_forward.0 | ap_int<3> x ap_int<3> -> impl=fabric
// LSQ thd=+-3 (b_w=2), HLS type=ap_int<3>
#pragma once
#include "ap_int.h"

typedef ap_int<3>    wi_layers_2_feed_forward_0_t;
typedef ap_int<3>    ai_layers_2_feed_forward_0_t;
typedef ap_int<16>        acc_layers_2_feed_forward_0_t;

#define LAYERS_2_FEED_FORWARD_0_OUT  768
#define LAYERS_2_FEED_FORWARD_0_IN   192
#define LAYERS_2_FEED_FORWARD_0_IMPL "fabric"

extern const wi_layers_2_feed_forward_0_t layers_2_feed_forward_0_W[768][192];
extern const float        layers_2_feed_forward_0_S[768];
extern const float        layers_2_feed_forward_0_B[768];