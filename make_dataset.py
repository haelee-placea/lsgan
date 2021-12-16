from glob import glob
import json
import numpy as np
import random


class MakeDataset():
    
    def __init__(self, dir_path, frames):
        pass
    
    
def make_dataset(dataset, false_ratio):
    
    st = random.randint(0, len(dataset)) 
    length = len(dataset) * false_ratio
    
    
    
    
    oints = list(map(lambda x: [x[0] - global_joint[0] ,x[1]-global_joint[1],x[2]-global_joint[2]] , joints))
    
    
    return n_data

def save_dataset():
    pass
      
def main():
    
    n_dataset = 100
    min_ratio, max_ratio = 0.1, 0.4 
    
    i = 0
    while i == n_dataset:
        i += 1
        
        tmp = np.random.random_sample((512, 19 , 3))
        false_ratio = random.uniform(min_ratio,max_ratio)
        print("false_ratio : ", false_ratio)
        
        make_dataset(tmp,false_ratio)
        save_dataset()
        
        
        

if __name__=='__name__':
    main()