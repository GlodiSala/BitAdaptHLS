// Delayed Scaling | embedding | ap_int<2> x ap_int<4> -> impl=fabric
#pragma once
#include "ap_int.h"

typedef ap_int<2>    wi_embedding_t;
typedef ap_int<4>    ai_embedding_t;
typedef ap_int<16>     acc_embedding_t;

#define EMBEDDING_OUT  128
#define EMBEDDING_IN   128
#define EMBEDDING_IMPL "fabric"

extern const wi_embedding_t embedding_W[128][128];
extern const float       embedding_S[128];
extern const float       embedding_B[128];