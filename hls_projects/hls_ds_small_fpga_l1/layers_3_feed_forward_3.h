// Delayed Scaling | layers.3.feed_forward.3 | ap_int<3> x ap_int<5> -> impl=fabric
#pragma once
#include "ap_int.h"

typedef ap_int<3>    wi_layers_3_feed_forward_3_t;
typedef ap_int<5>    ai_layers_3_feed_forward_3_t;
typedef ap_int<18>     acc_layers_3_feed_forward_3_t;

#define LAYERS_3_FEED_FORWARD_3_OUT  128
#define LAYERS_3_FEED_FORWARD_3_IN   512
#define LAYERS_3_FEED_FORWARD_3_IMPL "fabric"

extern const wi_layers_3_feed_forward_3_t layers_3_feed_forward_3_W[128][512];
extern const float       layers_3_feed_forward_3_S[128];
extern const float       layers_3_feed_forward_3_B[128];