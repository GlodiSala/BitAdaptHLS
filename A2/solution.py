import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
import copy
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict
import os
from common.utils import evaluate


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#                      ! DO NOT MODIFY THESE FUNCTIONS !
def conv_layer_generator(base_path: str = 'layer_prob_base.yaml', in_channels: int = 3, out_channels: int = 16,
                         kernel_size: int = 3, stride: int = 1, Height: int = 32, Width: int = 32,
                         save_path: str = 'conv1'):
    """
    Generate the yaml file for the conv layer
    Args:
        base_path: the path of the base yaml file
        in_channels: the number of input channels
        out_channels: the number of output channels
        kernel_size: the size of the kernel
        stride: the stride of the conv layer
        Height: the height of the input
        Width: the width of the input
        save_path: the path to save the generated yaml file

    """
    with open(base_path, 'r') as file:
        txt = file.read()
        txt_new = txt.replace('$in_channels$', str(in_channels))
        txt_new = txt_new.replace('$out_channels$', str(out_channels))
        txt_new = txt_new.replace('$kernel_size$', str(kernel_size))
        txt_new = txt_new.replace('$stride$', str(stride))
        txt_new = txt_new.replace('$Height$', str(Height))
        txt_new = txt_new.replace('$Width$', str(Width))
        with open(save_path+'.yaml', 'w') as output:
            output.write(txt_new)


def input_activation_hook(model, data):
    # add hook to record the min max value of the activation
    input_activation = {}
    output_activation = {}

    def add_range_recoder_hook(model):
        import functools
        def _record_range(self, x, y, module_name):
            x = x[0]
            input_activation[module_name] = x.detach()
            output_activation[module_name] = y.detach()

        all_hooks = []
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU)):
                all_hooks.append(m.register_forward_hook(
                    functools.partial(_record_range, module_name=name)))
        return all_hooks

    hooks = add_range_recoder_hook(model)
    model(data)

    # remove hooks
    for h in hooks:
        h.remove()
    return input_activation, output_activation


def Extract_Stats(path="timeloop-mapper.stats.txt"):
    """
    Extract the stats from timeloop-mapper.stats.txt
    Args:
        path: path to the stats file

    Returns: a dictionary of stats

    """
    mylines = []
    with open(path, 'rt') as myfile:
        for myline in myfile:
            if myline.find("Energy:") != -1:
                energy_total = float(myline[myline.find(":") + 2:myline.find("uJ")])
            elif myline.find("Cycles:") != -1:
                Cycles = int(myline[myline.find(":") + 2:])
            elif myline.find("EDP(J*cycle):") != -1:
                EDAP = float(myline[myline.find(":") + 2:])
            elif myline.find("GFLOPs (@1GHz):") != -1:
                GFLOPs = float(myline[myline.find(":") + 2:])

    return energy_total, Cycles, EDAP, GFLOPs

def Run_Accelergy(current_path=None, path_to_eyeriss_files='Q3'):
    #current path where timeloop generates the stat files
    if current_path is None : 
        current_path = os.getcwd()
    path_to_eyeriss_files = os.path.join(current_path, path_to_eyeriss_files)
    name_layers = os.listdir(os.path.join(path_to_eyeriss_files, 'prob'))
    os.system(f"rm -rf {current_path}/timeloop-model.stats.txt")
    energy_total = 0
    for l in name_layers:
        command = f"timeloop-mapper {path_to_eyeriss_files}/prob/{l} {path_to_eyeriss_files}/arch/components/*.yaml  " \
                  f"{path_to_eyeriss_files}/arch/eyeriss_like.yaml {path_to_eyeriss_files}/constraints/*.yaml  {path_to_eyeriss_files}/mapper/mapper.yaml" + " >/dev/null 2>&1"
        os.system(command)
        energy = Extract_Stats(path=f"{current_path}/timeloop-mapper.stats.txt")[0]
        print(f'energy consumption for {l[:-5]} : {energy} uJ')
        energy_total += energy
        os.system(f"rm -rf {current_path}/timeloop-model.stats.txt")
    print(f"Total energy consumption for ResNet-32: {energy_total} uJ")
    return energy_total

def generate_resnet_layers(model, base_path='common/layer_prob_base.yaml',  path='Q3/prob'):
    """
    Generate the yaml file for the conv layer
    Args:
        path: target path
        model: the model
        base_path: the path of the base yaml file

    Returns:

    """
    data = torch.rand(1, 3, 32, 32).to(model.conv1.weight.device)
    input_activation, output_activation = input_activation_hook(model, data)

    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        os.system(f"rm -rf {path}/*")
    start = True
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if hasattr(m, 'weight_mask'):
                num_zero_channels = torch.sum(torch.sum(torch.sum(torch.sum(model.state_dict()[name+"."+'weight_mask'], dim = -1), dim = -1), dim = -1) == 0).item()
            else:
                num_zero_channels = 0
            if start == True : 
              out_channel_last = m.in_channels
            conv_layer_generator(base_path = base_path, in_channels = out_channel_last, out_channels = m.out_channels - num_zero_channels, kernel_size = m.kernel_size[0], 
            stride= m.stride[0], Height = input_activation[name].shape[-2], Width = input_activation[name].shape[-1], save_path = path + "/" + name)

            start = False
            out_channel_last = m.out_channels - num_zero_channels

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def distill_from_frozen_teacher(
    teacher: nn.Module,
    student: nn.Module,
    train_loader,
    test_loader=None,
    *,
    epochs: int = 200,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    temperature: float = 2.0,   # T
    alpha_soft: float = 0.5,    # weight for soft targets; hard weight is (1-alpha_soft)
    device: torch.device = torch.device("cuda"),
):
    """
    Distill CIFAR-10 teacher -> scratch student.

    Loss:
      L = (1-a)*CE(student, y) + a*(T^2)*KL( softmax(t/T) || softmax(s/T) )
    """
    # teacher frozen
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student.train()

    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss definition
    ##### WRITE CODE HERE #####
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')


    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        student.train()
        total, correct, loss_sum = 0, 0, 0.0

        for x_t, x_s, y in train_loader:
            x_t = x_t.to(device)
            x_s = x_s.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            s_logits = student(x_s)
            with torch.no_grad():
                t_logits = teacher(x_t)

            # hard loss
            ##### WRITE CODE HERE #####
            loss_hard = ce_loss(s_logits, y)

            # soft loss
            ##### WRITE CODE HERE #####
            soft_t = F.softmax(t_logits / temperature, dim=1)
            soft_s = F.log_softmax(s_logits / temperature, dim=1)
            loss_soft = (temperature ** 2) * kl_loss(soft_s, soft_t)


            # total loss
            ##### WRITE CODE HERE #####
            loss = (1 - alpha_soft) * loss_hard + alpha_soft * loss_soft

            loss.backward()
            optimizer.step()

            bs = x_s.size(0)
            loss_sum += loss.item() * bs
            total += bs
            correct += (s_logits.argmax(1) == y).sum().item()

        scheduler.step()

        train_loss = loss_sum / max(total, 1)
        print("LOSS", train_loss)
        train_acc = correct / max(total, 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if test_loader is not None:
            test_acc = evaluate(student, test_loader, device)
            print(test_acc)
            history["test_acc"].append(test_acc)

    return student, history
