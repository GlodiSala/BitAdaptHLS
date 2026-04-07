import torch


class network_arch_builder(object):
    def __init__(self, args):
        self.args = args

    def optimizer(self, network_dnn):
        if self.args.optim_name == "AdamW":
            return torch.optim.AdamW(
                network_dnn.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.wd,
                amsgrad=True,
            )
        elif self.args.optim_name == "Adam":
            return torch.optim.Adam(
                network_dnn.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.wd,
            )
        elif self.args.optim_name == "SGD":
            return torch.optim.SGD(
                network_dnn.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.wd,
            )
        elif self.args.optim_name == "Adamax":
            return torch.optim.Adamax(
                network_dnn.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.wd,
            )
        elif self.args.optim_name == "RMSprop":
            return torch.optim.RMSprop(
                network_dnn.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.wd,
                momentum=0.95,
                centered=True,
            )
        elif self.args.optim_name == "RAdam":
            return torch.optim.RAdam(
                network_dnn.parameters(),
                lr=self.args.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.args.wd,
            )
