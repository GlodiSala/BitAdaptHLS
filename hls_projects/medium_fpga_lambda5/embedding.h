// Delayed Scaling | embedding | ap_int<3> x ap_int<6> -> impl=fabric
// LSQ thd=+-3 (b_w=2), HLS type=ap_int<3>
#pragma once
#include "ap_int.h"

typedef ap_int<3>    wi_embedding_t;
typedef ap_int<6>    ai_embedding_t;
typedef ap_int<17>        acc_embedding_t;

#define EMBEDDING_OUT  192
#define EMBEDDING_IN   128
#define EMBEDDING_IMPL "fabric"

extern const wi_embedding_t embedding_W[192][128];
extern const float        embedding_S[192];
extern const float        embedding_B[192];