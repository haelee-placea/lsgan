import argparse
import json

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import collections as matcoll
import numpy as np
import shutil

import os

def main(_args):


    f = open("data.txt", 'r')

    prev_clipname =""
    while True:
        line = f.readline()
        if not line:
            break

        metadata = line.split('\t')[:-1]
        clip_name = metadata[0]
        frame_valid = metadata[1:]
        folder_path = os.path.join(_args.input_folder, clip_name)
        print(clip_name)
        if prev_clipname != clip_name.split('/')[0]:
            folder_count = 0
            prev_clipname = clip_name.split('/')[0]
        file_list = os.listdir(folder_path)
        file_list.sort()

        start_num = int(file_list[0][6:12])
        iter_start = 0
        iter_end = 0

        prev_valid = frame_valid[0]


        for i in range(len(frame_valid)):
            if prev_valid == '0' and frame_valid[i] == '0':  # 계속 별로일 때
                a = 1
            elif prev_valid == '1' and frame_valid[i] == '1':  # 계속 잘될 때
                iter_end = i
            elif prev_valid == '0' and frame_valid[i] == '1':  # 잘되는게 시작될 때
                iter_start = i
                iter_end = i
            elif prev_valid == '1' and frame_valid[i] == '0':  # 별로가 시작될 때

                if int(iter_end) - int(iter_start) + 1 > 512:

                    print(f'{start_num + iter_start} {start_num + iter_end} {int(iter_end) - int(iter_start) + 1}')
                    folder_count += 1
                    to_folder = os.path.join(_args.output_folder, clip_name.split('/')[0], str(folder_count))
                    for j in range(start_num + iter_start, start_num + iter_end + 1):
                        from_path = os.path.join(folder_path, f'frame_{str(j).zfill(6)}.json')
                        to_path = os.path.join(to_folder, f'frame_{str(j).zfill(6)}.json')

                        if not os.path.exists(to_folder):
                            os.makedirs(to_folder)

                        shutil.copy(from_path, to_path)



            prev_valid = frame_valid[i]


    f.close()


    print("end")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-input-folder", "-input-folder", type=str, default="./my_panoptic2/annotations/171204_pose1", help="input frame folder name")
    parser.add_argument("-output-folder", "-output-folder", type=str, default="./my_panoptic2/annotations/171204_pose_seg", help="output frame folder name")

    args = parser.parse_args()
    main(args)
