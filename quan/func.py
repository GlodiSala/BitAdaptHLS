import torch as t
import torch.nn as nn
from .func import *
import torch.nn.functional as F

class QuanConv2d(t.nn.Conv2d):
    def __init__(self, m: t.nn.Conv2d, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == t.nn.Conv2d
        super().__init__(
            m.in_channels,
            m.out_channels,
            m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=True if m.bias is not None else False,
            padding_mode=m.padding_mode,
        )
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return self._conv_forward(quantized_act, quantized_weight, None)


class QuanLinear(t.nn.Linear):
    is_quantized_linear = True  # Flag to indicate this is a quantized linear layer
    def __init__(self, m: t.nn.Linear, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == t.nn.Linear
        super().__init__(
            m.in_features, m.out_features, bias=True if m.bias is not None else False
        )
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        """"
        quantized_act = quantized_act.view(quantized_act.size(0), -1) ######## just for JEROME's Work
        """
        return t.nn.functional.linear(quantized_act, quantized_weight, self.bias)
    
# ---------------------------------------------------------------------
class QuanMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, m: nn.MultiheadAttention, quan_w_fn=None, quan_a_fn=None):
        assert isinstance(m, nn.MultiheadAttention)
        super().__init__(
            embed_dim=m.embed_dim, num_heads=m.num_heads, dropout=m.dropout,
            bias=m.in_proj_bias is not None, add_bias_kv=m.bias_k is not None,
            add_zero_attn=m.add_zero_attn, kdim=m.kdim, vdim=m.vdim,
            batch_first=m.batch_first,
        )
        import copy
        self.quan_w_fn     = quan_w_fn                   # in_proj  [3d, d]
        self.quan_w_out_fn = copy.deepcopy(quan_w_fn)    # out_proj [d,  d]
        self.quan_a_fn     = quan_a_fn

        with t.no_grad():
            self.in_proj_weight.copy_(m.in_proj_weight)
            if m.in_proj_bias is not None:
                self.in_proj_bias.copy_(m.in_proj_bias)
            self.out_proj.weight.copy_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                self.out_proj.bias.copy_(m.out_proj.bias)

        if self.quan_w_fn is not None:
            self.quan_w_fn.init_from(self.in_proj_weight)
        if self.quan_w_out_fn is not None:
            self.quan_w_out_fn.init_from(self.out_proj.weight)

    def _quantise_params(self):
        W_in  = self.quan_w_fn(self.in_proj_weight)      if self.quan_w_fn     else self.in_proj_weight
        W_out = self.quan_w_out_fn(self.out_proj.weight) if self.quan_w_out_fn else self.out_proj.weight
        return W_in, W_out

    def forward(self, query, key, value, **kwargs):
        if self.quan_a_fn is not None:
            query = self.quan_a_fn(query)
            key   = self.quan_a_fn(key)
            value = self.quan_a_fn(value)
        W_in, W_out = self._quantise_params()
        attn_output, attn_weights = F.multi_head_attention_forward(
            query, key, value,
            embed_dim_to_check=self.embed_dim, num_heads=self.num_heads,
            in_proj_weight=W_in, in_proj_bias=self.in_proj_bias,
            bias_k=self.bias_k, bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn, dropout_p=self.dropout,
            out_proj_weight=W_out, out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=kwargs.get("key_padding_mask", None),
            need_weights=kwargs.get("need_weights", True),
            attn_mask=kwargs.get("attn_mask", None),
            use_separate_proj_weight=False,
            static_k=kwargs.get("static_k", None),
            static_v=kwargs.get("static_v", None),
        )
        return attn_output, attn_weights

QuanModuleMapping = {t.nn.Conv2d: QuanConv2d, t.nn.Linear: QuanLinear, 
                     t.nn.MultiheadAttention: QuanMultiheadAttention}