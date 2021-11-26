from glob import glob
import os
import argparse
import random
import json
import os

from torch.nn.modules.container import T

def read_json(path):
    
    with open(path) as f:
        json_data = json.load(f)
        
    return json_data

def save_json(path,file,data):
    makedirs(path)    
    with open(path + '/' + file,'w') as f:
        json.dump(data,f) 

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=False)
        
def split_folder_by_person(clip_path):
    annotes = sorted(glob(os.path.join(args.dataset,'annotations',clip_path) + '/*.json'))
    
    for json_path in annotes:
        folder, file = os.path.split(json_path)
        n_folder = folder.replace('panoptic',"n_panoptic")
        
        json_data = read_json(json_path)
        for d in json_data:
            save_path = n_folder + '/' + str(d['id']) 
            save_json(save_path,file,d)
            
                 
            

def main(args):
    
    tr_paths = read_json(os.path.join(args.dataset, args.train))
    val_path = os.path.join(args.dataset, args.valid)
    test_path = os.path.join(args.dataset, args.test)
     
    # max_id = 0
    for clip in tr_paths:
        split_folder_by_person(clip) 
    
    for clip in val_path:
        split_folder_by_person(clip) 
    
    for clip in test_path:
        split_folder_by_person(clip)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='dataset setting')
    parser.add_argument('--dataset',type=str, default='panoptic')
    parser.add_argument('--train',type=str, default='train_list.json')
    parser.add_argument('--valid',type=str, default='valid_list.json')
    parser.add_argument('--test',type=str, default='test_list.json')
    
    args = parser.parse_args()
    main(args)
    
    
    
    preprocess_data(dataset_path)
        