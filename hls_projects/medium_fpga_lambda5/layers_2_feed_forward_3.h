// Delayed Scaling | layers.2.feed_forward.3 | ap_int<3> x ap_int<4> -> impl=fabric
// LSQ thd=+-3 (b_w=2), HLS type=ap_int<3>
#pragma once
#include "ap_int.h"

typedef ap_int<3>    wi_layers_2_feed_forward_3_t;
typedef ap_int<4>    ai_layers_2_feed_forward_3_t;
typedef ap_int<17>        acc_layers_2_feed_forward_3_t;

#define LAYERS_2_FEED_FORWARD_3_OUT  192
#define LAYERS_2_FEED_FORWARD_3_IN   768
#define LAYERS_2_FEED_FORWARD_3_IMPL "fabric"

extern const wi_layers_2_feed_forward_3_t layers_2_feed_forward_3_W[192][768];
extern const float        layers_2_feed_forward_3_S[192];
extern const float        layers_2_feed_forward_3_B[192];