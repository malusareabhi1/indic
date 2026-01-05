# dataset.py
import cv2
import torch
from torch.utils.data import Dataset

class HandwrittenDataset(Dataset):
    def __init__(self, data, char2idx):
        self.data = data
        self.char2idx = char2idx

    def __len__(self):
        return len(self.data)

    def encode(self, text):
        return [self.char2idx[c] for c in text]

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (128, 32))
        img = img / 255.0

        img = torch.tensor(img).unsqueeze(0).float()
        label = torch.tensor(self.encode(label))

        return img, label
