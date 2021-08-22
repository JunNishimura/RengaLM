import torch
from torch.utils.data import Dataset

class RengaDataset(Dataset):
    def __init__(self, ids_list):
        super().__init__()
        self.data_length = len(ids_list)
        self.input_data = [ids[:-1] for ids in ids_list]
        self.target_data = [ids[1:] for ids in ids_list]

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        return torch.LongTensor(self.input_data[index]), torch.LongTensor(self.target_data[index])