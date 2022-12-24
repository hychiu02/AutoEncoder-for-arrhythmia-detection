import os
import torch

from torch.utils.data import Dataset


class VA_Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.fnames = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        input_data = []
        ground_truth = []
        with open(self.fnames[idx]) as f:
            for line in f.readlines():
                input_data.append(float(line))

        class_name = self.fnames[idx].split('/')[-1]
        class_name = class_name.split('-')[0]

        if((class_name=="VT") or (class_name=="VFb") or (class_name=="VFt")):
            ground_truth.append(1)
        elif((class_name=="AFb") or (class_name=="AFt") or (class_name=="SR") or (class_name=="SVT") or (class_name=="VPD")):
            ground_truth.append(0)
        else:
            raise NotImplementedError(f"Unknown class name: {class_name}")

        return torch.FloatTensor(input_data), torch.FloatTensor(ground_truth)
