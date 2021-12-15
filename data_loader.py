import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import json
import random
import pickle
import numpy as np
from PIL import Image
from glob import glob

random.seed(42)

def preprocess(dict_data,is_transform,is_mask,global_kpt = 'Pelvis'):
    joints = dict_data['kpt3d_body']
    mask = dict_data['kpt3d_body_mask']
     
    if not is_transform:
        return [[float(i),*j] for i,j in zip(mask,joints)]

    else:
        # subtract global key
        del_idx = dict_data['kpt_list'].index(global_kpt)
        global_joint = joints[del_idx]
        del joints[del_idx]
        del mask[del_idx]
        
        joints = list(map(lambda x: [x[0] - global_joint[0] ,x[1]-global_joint[1],x[2]-global_joint[2]] , joints))
        # return [[float(i),*j] for i,j in zip(mask,joints)]
        return joints
        

class PanopticDataset(data.Dataset):
    def __init__(self, dataset, json_path, input_frame=1024,global_kpt='Pelvis', is_discriminator=False):
        """[summary]
        Args:
            root ([type]): [description]
            json ([type]): [description]
            transform ([type], optional): [description]. Defaults to None.
        """
    
        self.dataset = dataset
        self.input_frame = input_frame 
        self.global_kpt = global_kpt
        
        self.is_discriminator = is_discriminator
        with open(os.path.join(self.dataset,json_path)) as f:
            json_data = json.load(f) # train/valid/test json file list
        
        self.clips = []
        for file in json_data:        
            self.clips.extend(glob(file + "/*"))

    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, index):
    
        selected_clip = self.clips[index]
        frames = glob(selected_clip + '/*.json')
        n_frames = len(frames)
        random_frame = random.randint(0,n_frames-self.input_frame)
        selected_frames = frames[random_frame : random_frame + self.input_frame]
         
        G_input_data = []
        D_input_data = []

        for fr in selected_frames:
            with open(fr) as f:
                data = json.load(f)
                assert self.global_kpt in data['kpt_list'], 'global key point is not in the list'
                G_input_data.append(preprocess(dict_data=data,is_transform=False,is_mask=True,global_kpt=self.global_kpt))
                D_input_data.append(preprocess(dict_data=data,is_transform=True,is_mask=False,global_kpt=self.global_kpt))
                
                
        return np.reshape(G_input_data,newshape=(self.input_frame,-1)), np.reshape(D_input_data,newshape=(self.input_frame,-1))

    def get_frame_idx(file):
        pass
        # ! 'panoptic/anootations/170407_haggling_a3/00_07'


if __name__ == "__main__":
        
    dataset = PanopticDataset(dataset='my_panoptic',json_path='test_list.json',input_frame = 1024)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    for data in dataloader:
        print(data)