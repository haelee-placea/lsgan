import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import json
import pickle
import numpy as np
from PIL import Image
from glob import glob


class PanoticDataset(data.Dataset):
    def __init__(self, dataset, json_path, transform=None):
        """[summary]
        Args:
            root ([type]): [description]
            json ([type]): [description]
            transform ([type], optional): [description]. Defaults to None.
        """
        self.dataset = dataset
        
        with open(os.path.join(self.dataset,json_path)) as f:
            json_data = json.load(f)
        
        for file in json_data:        
            self.annotates = glob(
                os.path.join(self.dataset, "annotations", file) + "/*.json", recursive=True
            )
            self.imgs = glob(
                os.path.join(self.dataset, "images", file) + "/*.jpg", recursive=True
            )
    
    def __len__(self):
        return len(self.annotates)
    

    def __getitem__(self, index): 
        # file = self.annotates[index]
        # annotate = json.load(open(self.annotates[index]))[0]['kpt3d_body']
        # mask = json.load(open(self.annotates[index]))[0]['kpt2d_body_mask']
        # img = np.array(Image.open(self.imgs[index]))
        data = np.random.rand(2048,4 * 18 )
        data[:,1] = np.round(data[:,0])
        return data

    def get_frame_dix(file):
        pass
        # ! 'panoptic/anootations/170407_haggling_a3/00_07'
        
        

# def get_loader(root,json):
# data_loader = torch.utils.data.DataLoader()

# PanoticDataset(dataset="panotic", json="train_list.json")

# for ann, img in PanoticDataset(dataset="panotic", json="train_list.json"):
#     print(ann,img)