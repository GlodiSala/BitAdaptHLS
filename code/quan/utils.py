import logging
import torch as t
import torch.nn as nn
from .func import *
from .quantizer import *
from quan.quantizer.lsq import LsqQuan
from quan.quantizer.quantizer import Quantizer  

def quantizer(default_cfg, this_cfg=None):
    target_cfg = dict(default_cfg)
    if this_cfg is not None:
        for k, v in this_cfg.items():
            target_cfg[k] = v

    if target_cfg["bit"] is None:
        q = IdentityQuan
    elif target_cfg["mode"] == "lsq":
        q = LsqQuan
    else:
        raise ValueError("Cannot find quantizer `%s`", target_cfg["mode"])

    target_cfg.pop("mode")
    return q(**target_cfg)


def find_modules_to_quantize(model, quan_scheduler):
    replaced_modules = {}
    quantizers       = {}
    IMP_DEFAULT      = 1 / 32 / 6
    scheduled_exc    = quan_scheduler.excepts or {}

    def has_wrapped_ancestor(qual_name: str) -> bool:
        parts = qual_name.split(".")
        for i in range(1, len(parts)):
            if ".".join(parts[:i]) in replaced_modules:
                return True
        return False

    for qual_name, module in model.named_modules():
        if has_wrapped_ancestor(qual_name):
            continue
        wrapper_cls = QuanModuleMapping.get(type(module), None)
        if wrapper_cls is None:
            continue

        q_w = LsqQuan(bit=8, min_bit=1, max_bit=8)
        q_a = LsqQuan(bit=8, min_bit=2, max_bit=8, all_positive=False)

        if qual_name in scheduled_exc:
            layer_cfg  = scheduled_exc[qual_name]
            weight_cfg = layer_cfg.get("weight", {})
            act_cfg    = layer_cfg.get("act",    {})
            q_w = quantizer(quan_scheduler.weight, weight_cfg)
            q_a = quantizer(quan_scheduler.act,    act_cfg)

        replaced_modules[qual_name] = wrapper_cls(
            module, quan_w_fn=q_w, quan_a_fn=q_a
        )
        quantizers[qual_name] = {
            "weight":     q_w,
            "activation": q_a,
            "imp":        IMP_DEFAULT,
        }

    return replaced_modules, quantizers


def fix_quantizer_references(model, quantizers):
    for qual_name, module in model.named_modules():
        if qual_name not in quantizers:
            continue
        if isinstance(module, QuanMultiheadAttention):
            quantizers[qual_name]["weight"]      = module.quan_w_fn
            quantizers[qual_name]["weight_out"]  = module.quan_w_out_fn  # ← ajout
            quantizers[qual_name]["activation"]  = module.quan_a_fn
        elif hasattr(module, 'quan_w_fn'):
            quantizers[qual_name]["weight"]      = module.quan_w_fn
            quantizers[qual_name]["activation"]  = module.quan_a_fn
    return quantizers

def calibrate_activation_quantizers(model, quantizers, data_loader, device, num_batches=10):
    model = model.to(device)
    model.eval()
    hooks = []
    activation_stats = {}

    for qual_name, module in model.named_modules():
        if qual_name not in quantizers:
            continue
        def make_hook(name):
            def hook(mod, inp, out):
                x = inp[0].detach()
                if name not in activation_stats:
                    activation_stats[name] = []
                activation_stats[name].append(x.abs().mean().item())
            return hook
        hooks.append(module.register_forward_hook(make_hook(qual_name)))

    with t.no_grad():
        for i, (channel, channel_norm) in enumerate(data_loader):
            if i >= num_batches:
                break
            model(channel_norm.float().to(device))

    for h in hooks:
        h.remove()

    for qual_name, q_dict in quantizers.items():
        if qual_name in activation_stats and hasattr(q_dict['activation'], 's'):
            mean_abs = sum(activation_stats[qual_name]) / len(activation_stats[qual_name])
            thd_pos = 2 ** int(q_dict['activation'].bit.item()) - 1
            q_dict['activation'].s = t.nn.Parameter(
                t.tensor(mean_abs * 2 / (thd_pos ** 0.5))
            )

    model.train()
    return quantizers



def replace_module_by_names(model, modules_to_replace):
    def helper(child: t.nn.Module):
        for n, c in child.named_children():
            if type(c) in QuanModuleMapping.keys():
                for full_name, m in model.named_modules():
                    if c is m:
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)
    helper(model)
    return model