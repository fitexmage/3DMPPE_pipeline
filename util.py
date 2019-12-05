import pickle
import numpy as np
import csv

from config import cfg as pipeline_cfg

def get_frames(video):
    frames = []
    while video.isOpened():
        success, frame = video.read()
        if success:
            frames.append(frame)
        else:
            return frames


def load(binary_path):
    with open(binary_path, "rb") as f:
        data = pickle.load(f)
    return data


def get_header():
    header_list = []
    for i in range(pipeline_cfg.joint_num):
        header_list.extend([str(i)+"_x", str(i)+"_y", str(i)+"_z"])
    return header_list


def compute_distance(prev_pos, cur_pos):
    return np.sum((cur_pos - prev_pos) ** 2)


def get_row(data, prev_pos):
    pos_list = []
    if prev_pos is None:
        data = data[0]
    else:
        data = data[np.argmin(np.array([compute_distance(prev_pos, person[pipeline_cfg.spine_inx]) for person in data]))]

    for joint in data:
        for pos in joint:
            pos_list.append(str(pos))
    return pos_list, data[pipeline_cfg.spine_inx]


def to_csv(binary_path, csv_path):
    data = load(binary_path)
    print(len(data))
    print(data[0].shape)

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        writer.writerow(get_header())
        prev_pos = None
        for row in data:
            pos_list, prev_pos = get_row(row, prev_pos)
            writer.writerow(pos_list)