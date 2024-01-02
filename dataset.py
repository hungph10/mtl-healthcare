import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ACCDataset(Dataset):
    def __init__(self, features, cls_target, reg_target):
        super().__init__()
        self.x = features
        self.y_cls = cls_target
        self.y_reg = reg_target

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        feature = self.x[idx]
        cls_target = self.y_cls[idx]
        reg_target = self.y_reg[idx]
        return feature.to(device), cls_target.to(device), reg_target.to(device)


