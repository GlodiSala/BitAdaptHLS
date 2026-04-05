import logging
import math
import time
import torch as t
from util import AverageMeter
from tqdm import tqdm

__all__ = ["train", "validate"]

logger = logging.getLogger()


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    lr_scheduler,
    epoch,
    monitors,
    args,
    args_,
    quantizers,
):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    # logger.info("Training: %d samples (%d per mini-batch)", total_sample, batch_size)
    model = model.to(args.device.type)

    model.train()
    # end_time = time.time()
    overall_time = overall_time_loss = 0
    for batch_idx, (channel, channel_norm) in enumerate(tqdm(train_loader, total=len(train_loader), mininterval=60.0)):
        inputs = channel_norm.float().to(args.device.type)
        channel = channel.to(args.device.type)
        # Set gradient to 0.
        optimizer.zero_grad()
        # Feed forward Reg
        model.train()
        

        if args_.BF_Sch in ["HBF"]:
            out_dp_R, out_dp_I, out_ap = model(inputs)
            # W calc
            W_out = (out_dp_R + 1j * out_dp_I).view(
                -1,
                args_.act_Usr,
                args_.Nr,
                args_.Nrf,
            )
            A_out = t.exp(1j * (out_ap)).view(-1, args_.Nt, args_.Nrf)

            loss = -1 * criterion(W_out, A_out, channel.to(args.device.type))

        elif args_.BF_Sch in ["FDP"]:
            if args_.method in ["cnn-based"]:
                F_out = model(inputs)
                # Criterion returns sum_rate - energy, negate for minimization
                loss = -1 * criterion(F_out, t.squeeze(channel), quantizers, model)

        # Energy efficiency loss
        total_loss = loss
        losses.update(total_loss.item(), inputs.size(0))
        acc1, acc5 = loss, loss
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        if lr_scheduler is not None:
            lr_scheduler.step(epoch=epoch, batch=batch_idx)
        
        total_loss.backward()
        
        t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # start = time.time()
        optimizer.step()
       
    return top1.avg, top5.avg, losses.avg


def validate(data_loader, model, criterion, epoch, monitors, args, args_):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    model = model.to(args.device.type)
    # logger.info("Validation: %d samples (%d per mini-batch)", total_sample, batch_size)

    model.eval()
    end_time = time.time()
    for batch_idx, (tchannel, tchannel_n) in enumerate(data_loader):

        with t.no_grad():
            test_inputs = tchannel_n.float().to(args.device.type)
            if args_.BF_Sch in ["HBF"]:
                # Forward pass reg
                pred_dp_R, pred_dp_I, pred_ap = model(test_inputs)

                # W calc
                pred_W_out = (pred_dp_R + 1j * pred_dp_I).view(
                    -1,
                    args_.act_Usr,
                    args_.Nr,
                    args_.Nrf,
                )
                pred_A_out = t.exp(1j * (pred_ap)).view(-1, args_.Nt, args_.Nrf)

                loss = criterion(
                    pred_W_out,
                    pred_A_out,
                    tchannel.to(args.device.type),
                )
            elif args_.BF_Sch in ["FDP"]:
                if args_.method in ["cnn-based"]:
                    
                    pred_F_out = model(test_inputs)
                    loss = criterion(pred_F_out, t.squeeze(tchannel.to(args.device.type)))

            acc1, acc5 = loss, loss
            losses.update(loss.item(), test_inputs.size(0))
            top1.update(acc1.item(), test_inputs.size(0))
            top5.update(acc5.item(), test_inputs.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (batch_idx + 1) % args.log.print_freq == 0:
                for m in monitors:
                    m.update(
                        epoch,
                        batch_idx + 1,
                        steps_per_epoch,
                        "Validation",
                        {
                            "Loss": losses,
                            "Top1": top1,
                            "Top5": top5,
                            "BatchTime": batch_time,
                        },
                    )

    return top1.avg, top5.avg, losses.avg