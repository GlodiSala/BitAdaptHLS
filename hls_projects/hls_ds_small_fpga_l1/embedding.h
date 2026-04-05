// Delayed Scaling | embedding | ap_int<3> x ap_int<6> -> impl=fabric
#pragma once
#include "ap_int.h"

typedef ap_int<3>    wi_embedding_t;
typedef ap_int<6>    ai_embedding_t;
typedef ap_int<17>     acc_embedding_t;

#define EMBEDDING_OUT  128
#define EMBEDDING_IN   128
#define EMBEDDING_IMPL "fabric"

extern const wi_embedding_t embedding_W[128][128];
extern const float       embedding_S[128];
extern const float       embedding_B[128];