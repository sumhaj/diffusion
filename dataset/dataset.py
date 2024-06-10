import os
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, image_path, label_file):
        self.image_path = image_path
        self.image_labels = pd.read_csv(label_file)

    def __len__(self):
        return len(self.image_labels)
        # f = open(self.label_file, 'r')
        # f.seek(0, 2)
        # return f.tell() 

    def __getitem__(self, index):
        image = read_image(os.path.join(self.image_path, self.image_labels.iloc[index, 0]))
        image = image.float()
        image = (image - 127.5) / 127.5
        # label = ToTensor()(self.image_labels.iloc[index, 1])
        label = torch.tensor(self.image_labels.iloc[index, 1])
        data = (image, label)
        return data

    # def __getitems__(self, index_list):
    #     image_list, label_list = [], []
    #     for index in index_list:
    #         image = read_image(os.path.join(self.image_path, self.image_labels.iloc[index, 0]))
    #         image = image.float()
    #         image = (image - 127.5) / 127.5
    #         # label = ToTensor()(self.image_labels.iloc[index, 1])
    #         label = torch.tensor(self.image_labels.iloc[index, 1])
    #         image_list.append(image)
    #         label_list.append(label)
    #     return image_list, label_list



        
        