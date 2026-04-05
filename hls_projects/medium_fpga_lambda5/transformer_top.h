// Delayed Scaling Transformer | S=4 T=128 D=192 NL=4 OUT=128
// All intermediate activations in float after first dequantization
#pragma once
#include "ap_fixed.h"
#include "ap_int.h"
#include "embedding.h"
#include "output.h"
#include "layers_0_feed_forward_0.h"
#include "layers_0_attention_in_proj.h"
#include "layers_0_norm1.h"
#include "layers_0_norm2.h"
#include "layers_1_feed_forward_0.h"
#include "layers_1_attention_in_proj.h"
#include "layers_1_norm1.h"
#include "layers_1_norm2.h"
#include "layers_2_feed_forward_0.h"
#include "layers_2_attention_in_proj.h"
#include "layers_2_norm1.h"
#include "layers_2_norm2.h"
#include "layers_3_feed_forward_0.h"
#include "layers_3_attention_in_proj.h"
#include "layers_3_norm1.h"
#include "layers_3_norm2.h"

#define NUM_LAYERS  4
#define TOKEN_DIM   128
#define EMB_DIM     192
#define OUTPUT_DIM  128
#define SEQ_LEN     4

typedef ap_fixed<16,2>   input_t;
// Note: intermediate buffers use float (post Delayed Scaling dequantization)
