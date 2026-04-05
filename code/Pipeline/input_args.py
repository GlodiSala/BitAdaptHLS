import argparse
import numpy as np
import datetime
import os


class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):

        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)


class input_args:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        model_name = self.generate_name()
        self.parser.add_argument(
            "--model_name",
            type=str,
            default=model_name,
            help="model name for saving the weights",
        )
        self.parser.add_argument(
            "--save_model",
            action="store_true",
            help="Save the quantized model after each epoch",
        )
        self.parser.add_argument(
            "--load_model",
            default=False,
            help="Load the pre-trained model weights!",
        )
        self.parser.add_argument(
            "--noLOG",
            default=False,
            help="disable logging into WandB!",
        )
        self.parser.add_argument(
            "--deploy",
            default=True,
            help="Load model and Fine-tune on deployment scenario!",
        )
        self.parser.add_argument(
            "--zeroshot",
            default=False,
            help="Get Zero shot performance!",
        )
        self.parser.add_argument(
            "--dir",
            type=str,
            default="/export/tmp/sala/Experiments/",
            help="Directory for saving models and input parameters",
        )

        self.parser.add_argument("--seed", type=float, default=5417, help="seed")
        self.parser.add_argument(
            "--device", type=str, default="cuda:0", help="Cuda device"
        )
        self.parser.add_argument(
            "--device_id",
            type=int,
            default=np.array([0]),
            help="Cuda device IDs",
            action=Store_as_array,
            nargs="+",
        )
        self.parser.add_argument(
            "--data_aug", action="store_true", help="Augment dataset"
        )
        self.parser.add_argument("--LOS", default=True, help="Use LOS users")
        self.parser.add_argument("--noisy", action="store_true", help="Use noisy CSI")
        self.parser.add_argument(
            "--channelType",
            type=str,
            default="matlab",
            help="Dataset type (deepMIMO/matlab/statistical)",
        )
        self.parser.add_argument("--act_Usr", type=int, default=4, help="Active users")
        self.parser.add_argument(
            "--BF_Sch", type=str, default="FDP", help="Beamforming Scheme (HBF/FDP)"
        )
        self.parser.add_argument(
            "--noise_pwr", type=float, default=1e-13, help="Noise Power"
        )
        self.parser.add_argument(
            "--datasetsize", type=int, default=int(1e6), help="Dataset size"
        )
        self.parser.add_argument(
            "--comm_sch", type=str, default="TDD", help="(TDD/FDD)"
        )
        self.parser.add_argument(
            "--bs_ant",
            type=int,
            default=np.array([1, 8, 8]),
            help="BS's antenna",
            action=Store_as_array,
            nargs="+",
        )
        self.parser.add_argument(
            "--ue_ant",
            type=int,
            default=np.array([1, 1, 1]),
            help="User's antenna",
            action=Store_as_array,
            nargs="+",
        )

        self.parser.add_argument(
            "--method",
            type=str,
            default="cnn-based",
            help="methods for BF (cnn-based/wmmse-based)",
        )

        self.initial_args, _ = self.parser.parse_known_args()

        if self.initial_args.BF_Sch == "HBF":
            self.parser.add_argument(
                "--Nrf", type=int, default=8, help="Number of Rf chains"
            )

        self.parser.add_argument(
            "--Nt",
            type=int,
            default=np.prod(self.initial_args.bs_ant),
            help="number of antennas element in BS",
        )
        self.parser.add_argument(
            "--Nr",
            type=int,
            default=np.prod(self.initial_args.ue_ant),
            help="number of antennas element in User",
        )
        if self.initial_args.channelType == "deepMIMO":
            self.inputs_deepMIMO()

        elif self.initial_args.channelType == "matlab":
            self.inputs_matlab()

        elif self.initial_args.channelType == "statistical":
            pass

        self.inputs_dnn()
        self.args = self.parser.parse_args()
        self.save_args_to_file(
            self.args.dir
            + self.args.model_name
            + "/"
            + self.args.model_name
            + "_args"
            + ".txt"
        )

    def inputs_deepMIMO(self):
        self.parser.add_argument(
            "--dir_dataset",
            type=str,
            default=r"/usr/datasets/deepMIMO/HH_channels/deepMIMO_v2/scenarios/updated/",
            help="Directory for loading datasets",
        )
        self.parser.add_argument(
            "--scenario", type=str, default="O1_28", help="DeepMIMO Scenario"
        )
        self.parser.add_argument(
            "--user_row_first", type=int, default=962 - 500, help="First row"
        )
        self.parser.add_argument(
            "--user_row_last", type=int, default=962 + 500, help="Last row"
        )
        self.parser.add_argument(
            "--act_BS",
            type=int,
            default=np.array([3]),
            help="Active BSs",
            action=Store_as_array,
            nargs="+",
        )
        self.parser.add_argument("--K", type=int, default=1, help="Number of SSB")
        self.parser.add_argument("--Npath", type=int, default=1, help="Number of paths")

    def inputs_Transformer(self):
        self.parser.add_argument(
            "--seq_length", type=int, default=4, help="Sequence length for Transformer"
        )
        self.parser.add_argument(
            "--token_dim", type=int, default=128, help="Token dimension for Transformer"
        )
        self.parser.add_argument(
            "--context_dim", type=int, default=2, help="Context dimension for Transformer"
        )
        self.parser.add_argument(
            "--embedding_dim",
            type=int,
            default=128,
            help="Embedding dimension for Transformer",
        )
        self.parser.add_argument(
            "--num_heads", type=int, default=2, help="Number of heads in Transformer"
        )
        self.parser.add_argument(
            "--hidden_dim", type=int, default=1024, help="Hidden dimension in Transformer"
        )
        self.parser.add_argument(
            "--dropout", type=float, default=0.05, help="Dropout rate in Transformer"
        )
        self.parser.add_argument(
            "--num_layers", type=int, default=4, help="Number of layers in Transformer"
        )

    def inputs_dnn(self):
        self.parser.add_argument(
            "--ratio",
            type=float,
            default=0.8,
            help="ratio of splitting train set and test set",
        )
        self.parser.add_argument(
            "--batch_size", type=int,
            default=int(os.environ.get('BATCH_SIZE', 1000)),
            help="batch size"
        )
        self.parser.add_argument(
            "--epoch_size", type=int, default=20, help="epoch size"
        )
        self.parser.add_argument(
            "--lr", type=float, default=0.0001, help="learning rate"
        )
        self.parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
        self.parser.add_argument(
            "--n_hidden",
            type=int,
            default=512,
            help="number of neurons in hidden layer",
        )

        if self.initial_args.method == "cnn-based":
            # Define possible channel sizes
            # channel_sizes = [128, 64, 32]
            
            # Add arguments dynamically based on channel sizes
            # for i, size in enumerate(channel_sizes):
            self.parser.add_argument(
                f"--out_channel", type=int, default=64, help=f"number of channels in CNN"
            )
                
            self.parser.add_argument(
                "--kernel_s", type=int, default=3, help="kernel size"
            )
            self.parser.add_argument("--padding", type=int, default=1, help="padding")
            self.parser.add_argument(
                "--in_ch", type=int, default=2, help="input channel"
            )
        else:
            self.parser.add_argument(
                "--single_model", default=True, help="single model or not"
            )

        self.parser.add_argument(
            "--p_dropout", type=float, default=0.01, help="pr of dropout"
        )
        self.parser.add_argument(
            "--optim_name", type=str, default="Adam", help="optimizer"
        )
    def inputs_matlab(self):
        self.parser.add_argument(
            "--dir_dataset",
            type=str,
            default=r"/usr/datasets/deepMIMO/HH_channels/Dataset_matlab/",
            help="Directory for loading datasets",
        )
        self.parser.add_argument(
            "--scenario",
            type=str,
            default=["stecath"],# "decarie", "ericsson", "stecath"
            nargs="+",
            help="DeepMIMO Scenario",
        )

    def generate_name(self):
        current_time = datetime.datetime.now()
        name = name = "Model_" + str(current_time.strftime("%Y%m%d%H%M%S%f"))
        return name

    def save_args_to_file(self, filename: str) -> None:
        if not os.path.isdir(self.args.dir + self.args.model_name + "/"):
            os.mkdir(self.args.dir + self.args.model_name + "/")
        with open(filename, "w") as f:
            for arg, value in vars(self.args).items():
                f.write(f"{arg}: {value}\n")