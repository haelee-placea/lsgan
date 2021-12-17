import argparse
import json

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import collections as matcoll
import numpy as np

import os

class PlotViewer3D:
    def __init__(self, fig, file_list, folder_path):
        self.fig = fig
        self.frame_idx = 0
        self.folder_path = folder_path
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='none')
        radius = 2000
        self.ax.set_xlim(radius, -radius)
        self.ax.set_ylim(radius, -radius)
        self.ax.set_zlim(radius, 0)
        self.ax.view_init(110., 90.)
        self.points_set = []
        self.connection_set = []
        self.cid = fig.canvas.mpl_connect('key_press_event', self)
        self.timer = fig.canvas.new_timer(interval=33)
        self.timer.add_callback(self.timer_callback)
        self.timer_status = False
        self.annot_data = file_list
        self.connections = [[0, 1], [0, 2], [0, 3], [0, 9],
                            [1, 17],[17, 18],[1, 15],[15, 16],
                            [2, 6], [6, 7], [7, 8],
                            [2, 12], [12, 13], [13, 14],
                            [9, 10], [10, 11],
                            [3, 4], [4, 5],
                            ]

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

    def __call__(self, event):
        if event:
            if event.key == 'a':
                self.frame_idx -= 1
            elif event.key == 'd':
                self.frame_idx += 1
            elif event.key == 'w':
                self.frame_idx -= 3
            elif event.key == 'e':
                self.frame_idx += 3
            elif event.key == 'ctrl+a':
                self.frame_idx -= 10
            elif event.key == 'ctrl+d':
                self.frame_idx += 10
            elif event.key == 'alt+a':
                self.frame_idx -= 100
            elif event.key == 'alt+d':
                self.frame_idx += 100
            elif event.key == 'home':
                self.frame_idx = 0
            elif event.key == ' ':
                if self.timer_status:
                    self.timer_status = False
                    self.timer.stop()
                else:
                    self.timer_status = True
                    self.timer.start()
            elif event.key == 'escape':
                plt.close()
                return
            else:
                return
        else:
            self.frame_idx = 0

        if self.frame_idx < 0:
            self.frame_idx = len(self.annot_data) - 1
        if self.frame_idx >= len(self.annot_data):
            self.frame_idx = 0

        self.draw()

    def timer_callback(self):
        self.frame_idx += 5
        if self.frame_idx >= len(self.annot_data):
            self.frame_idx = 0
        self.draw()

    def draw(self):
        self.ax.set_title(f'{self.frame_idx}/{self.annot_data[self.frame_idx]})', loc='right')

        for set in self.points_set:
            set.remove()
        for set in self.connection_set:
            self.ax.collections.remove(set)

        self.points_set = []
        self.connection_set = []
        color = 'blue'
        dot_size = 10

        # draw skeleton
        current_file_name = self.annot_data[self.frame_idx]

        with open(os.path.join(self.folder_path, current_file_name), "r") as st_json:
            raw = json.load(st_json)
            kpt3d = np.array(raw['kpt3d_body'])

        self.points_set.append(
            self.ax.scatter(kpt3d[:, 0], kpt3d[:, 1], kpt3d[:, 2], color=color, s=dot_size))

        lines = []

        for con in self.connections:
            p1 = kpt3d[con[0]]
            p2 = kpt3d[con[1]]
            lines.append([p1, p2])
        self.connection_set.append(Line3DCollection(lines, colors='gold'))

        for connection in self.connection_set:
            self.ax.add_collection3d(connection)

        plt.tight_layout()
        plt.draw()


def main(_args):
    clip_name = "00_03"
    id_name = "2"

    folder_path = os.path.join(_args.input_folder, clip_name, id_name)
    file_list = os.listdir(folder_path)
    file_list.sort()

    # visualize 3d points with matplotlib
    matplotlib.use('TkAgg')
    clicker = PlotViewer3D(plt.figure(figsize=(15, 15)), file_list=file_list, folder_path=folder_path)

    clicker.__call__(None)
    plt.show()

    print("end")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-input-folder", "-input-folder", type=str, default="./my_panoptic/annotations/171204_pose1", help="input frame folder name")

    args = parser.parse_args()
    main(args)
