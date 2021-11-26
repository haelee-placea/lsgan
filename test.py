import json


file = 'panoptic/annotations/170407_haggling_b3/00_01/frame_010922.json'

with open(file) as f:
    json_data = json.load(f)
    print(json_data)
    
    
    