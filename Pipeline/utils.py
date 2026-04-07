import torch
import torch.utils.data as data
from termcolor import colored
import os
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat as lmat


class Data_Reader(data.Dataset):
    def __init__(self, channel, channel_norm):
        self.channel_ = torch.tensor(channel)
        self.channel_norm_ = torch.tensor(channel_norm)
        self.n_samples = channel.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.channel_[index], self.channel_norm_[index]


class utils_(object):
    def __init__(self, args):
        self.args = args
        os.chdir(self.args.dir)
        self.device = torch.device(
            self.args.device if torch.cuda.is_available() else "cpu"
        )
        print(
            "Is Cuda available? ",
            (
                colored("True", "green")
                if torch.cuda.is_available()
                else colored("False", "red")
            ),
        )
        print("Which device?", colored(self.device, "cyan"))

    def Data_Load(self):
        if self.args.channelType == "matlab":
            mat_data = []
            for scenario in self.args.scenario:
                if scenario in ["decarie", "ericsson", "stecath"]:
                    for los in ["LOS", "NLOS"]:
                        print("dataset_" + scenario + "_" + los + "_raytracing.mat")
                        mat_data.append(
                            lmat(
                                self.args.dir_dataset
                                + "dataset_"
                                + scenario
                                + "_"
                                + los
                                + "_raytracing.mat"
                            )
                        )
                else:
                    print(
                        "dataset_"
                        + scenario
                        + ("_LOS" if self.args.LOS else "_NLOS")
                        + "_raytracing_Reflectaion10.mat"
                    )
                    mat_data.append(
                        lmat(
                            self.args.dir_dataset
                            + "dataset_"
                            + scenario
                            + ("_LOS" if self.args.LOS else "_NLOS")
                            + "_raytracing_Reflectaion10.mat"
                        )
                    )
            if not scenario in ["decarie", "ericsson", "stecath"]:
                size_scenario = self.args.datasetsize // len(self.args.scenario)
                for n_scenario in range(len(self.args.scenario)):
                    channel_dts = mat_data[n_scenario][
                        "CSIs" + ("_LOS" if self.args.LOS else "_NLOS")
                    ]
                    rnd_idx = np.random.choice(
                        np.arange(channel_dts.shape[0]),
                        size=(size_scenario, self.args.act_Usr),
                    )
                    slc_channel = (channel_dts[rnd_idx][:, :, np.newaxis, :]).astype(
                        "complex64"
                    )
                    self.slc_channel = (
                        slc_channel
                        if n_scenario == 0
                        else np.concatenate((self.slc_channel, slc_channel), axis=0)
                    )
                    locations = mat_data[n_scenario][
                        "Locations" + ("_LOS" if self.args.LOS else "_NLOS")
                    ][rnd_idx]
                    self.locations = (
                        locations
                        if n_scenario == 0
                        else np.concatenate((self.locations, locations), axis=0)
                    )
            else:
                channel_dts_los = mat_data[0]["CSIs" + "_LOS"]
                channel_dts_nlos = mat_data[1]["CSIs" + "_NLOS"]
                channel_dts = np.concatenate(
                    (channel_dts_los, channel_dts_nlos), axis=0
                )
                rnd_idx = np.random.choice(
                    np.arange(channel_dts.shape[0]),
                    size=(self.args.datasetsize, self.args.act_Usr),
                )
                slc_channel = (channel_dts[rnd_idx][:, :, np.newaxis, :]).astype(
                    "complex64"
                )
                self.slc_channel = slc_channel
                self.locations = 0

        elif self.args.channelType == "statistical":
            pass

        self.data_preprocess()

        DataBase = Data_Reader(self.slc_channel, self.slc_channel_norm)

        my_dataloader, my_testloader = self.data_loader(DataBase, self.args.ratio)

        return my_dataloader, my_testloader

    def data_loader(self, DataBase, ratio=0.85):
        # Dataset is fully in RAM → num_workers=0 is optimal
        num_workers = 0  # remove the env variable entirely, not needed
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            DataBase,
            [int(ratio * len(DataBase)), len(DataBase) - int(ratio * len(DataBase))],
        )
        print(colored("The size of DataSet: ", "yellow"), len(DataBase) / 1e6, "M")
        print(colored("The size of Training set: ", "yellow"), len(train_dataset) / 1e6, "M")
        print(colored("The size of Test set: ", "yellow"), len(test_dataset) / 1e6, "M")
        
        my_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,      # in-process, fastest for RAM datasets
            pin_memory=True,    # still helps for GPU transfer
        )
        my_testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        return my_dataloader, my_testloader

    def data_preprocess(self):
        if self.args.data_aug:
            channel = np.zeros(
                (int(1e6) - self.arg.datasetsize, 4, 1, self.slc_channel.shape[-1])
            )
            channel = np.concatenate((channel, self.slc_channel), axis=0)
            ind = np.random.randint(
                0,
                high=self.slc_channel.shape[0],
                size=(int(1e6) - self.args.datasetsize, 4),
                dtype=int,
            )
            ind_user = np.random.randint(
                0, high=4, size=(int(1e6) - self.args.datasetsize, 4), dtype=int
            )
            for i in tqdm(range(int(1e6) - self.args.datasetsize)):
                ch = self.slc_channel[ind[i], ind_user[i]]
                channel[i] = ch
            self.slc_channel = channel.astype("complex64")
        pilots = np.eye(self.args.act_Usr)
        r_sig_ = np.einsum("ijkl,jj->ijl", np.conjugate(self.slc_channel), pilots)
        self.r_sig_norm = np.zeros_like(r_sig_)
        if self.args.noisy:
            beta = 0.1
            r_sig_ = (
                np.sqrt(1 - beta**2) * r_sig_
                + (
                    np.random.normal(size=r_sig_.shape)
                    + 1j * np.random.normal(size=r_sig_.shape)
                )
                * self.args.noise_pwr
                * beta
            ).astype("complex64")
            self.slc_channel = np.expand_dims(r_sig_, axis=2)
        print(colored("Normalizing CSI", "cyan"), " ... \n")
        for i in tqdm(range(len(self.r_sig_norm))):
            cnst = np.sqrt(np.sum(np.power(abs(r_sig_[i, :, :]), 2)))
            self.r_sig_norm[i, :, :] = r_sig_[i, :, :] / cnst
        self.slc_channel_norm = np.concatenate(
            (
                # np.expand_dims(np.abs(r_sig_norm), 1),
                # np.expand_dims(np.angle(r_sig_norm), 1),
                np.expand_dims(np.imag(self.r_sig_norm), 1),
                np.expand_dims(np.real(self.r_sig_norm), 1),
            ),
            1,
        )
        print(colored("Creating the Dataset", "blue"), " Done! ")
