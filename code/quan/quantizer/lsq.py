import torch as t
from .quantizer import Quantizer

    
def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def floor_pass(n):
    n_ = n.floor()
    n_grad = n
    return (n_ - n_grad).detach() + n_grad

class LsqQuan_old(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = t.nn.Parameter(t.ones(1))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x


class LsqQuan(Quantizer):
    def __init__(self, bit, min_bit=1, max_bit=16, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)
        self.bit = t.nn.Parameter(t.tensor(float(bit)))  # Trainable bitlength
        self.min_bit = min_bit
        self.max_bit = max_bit
        self.all_positive = all_positive
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.s = t.nn.Parameter(t.ones(1))
        self.thd_pos = 2 ** bit - 1

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def quantization(self, x, num_bit):
        thd_pos = 2 ** num_bit - 1
        s_grad_scale = 1.0 / ((thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x_scaled = x / s_scale
        
        # Fix: ensure thd_pos is always a scalar Tensor for clamp compatibility
        thd_pos_val = thd_pos if isinstance(thd_pos, t.Tensor) else t.tensor(float(thd_pos))
        
        if self.all_positive:
            x_clamped = t.clamp(x_scaled, t.zeros_like(thd_pos_val), thd_pos_val)
        else:
            x_clamped = t.clamp(x_scaled, -thd_pos_val, thd_pos_val)
        
        x_rounded = round_pass(x_clamped)
        return x_rounded * s_scale

    def forward(self, x):
        self.bit.data = t.clamp(self.bit, self.min_bit, self.max_bit)
        b = floor_pass(self.bit)  # Integer part
        alpha = self.bit - b  # Fractional part
        b_next = t.clamp(b + 1, self.min_bit, self.max_bit)
        return (1 - alpha) * self.quantization(x, b) + alpha * self.quantization(x, b_next)