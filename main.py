import cv2
import time
import pickle
import csv
import numpy as np

from detectnet import get_detectnet_config, get_detectnet_model, get_image_bounding_boxes, get_frames
from rootnet import get_input, set_rootnet_config, get_rootnet_model, get_root
from posenet import set_posenet_config, get_posenet_model, get_pose
from config import cfg as pipeline_cfg
from vis import visualize

def main():
    detectnet_config = get_detectnet_config()
    detectnet_model = get_detectnet_model(detectnet_config)
    set_rootnet_config()
    rootnet_model = get_rootnet_model()
    set_posenet_config()
    print("A")
    posenet_model = get_posenet_model()
    print("A")

    video = cv2.VideoCapture(pipeline_cfg.input_video_path)

    # image = cv2.imread(pipeline_cfg.input_image_path)
    # from detectron2.engine import DefaultPredictor
    # person_boxes = get_image_bounding_boxes(image, DefaultPredictor(detectnet_config))
    # if len(person_boxes) == 0:
    #     return
    #
    # person_images, k_values = get_input(image, person_boxes)
    # rootnet_preds = get_root(image, person_boxes, rootnet_model, person_images, k_values)
    # # print(rootnet_preds)
    # posenet_preds = get_pose(image, person_boxes, posenet_model, person_images, rootnet_preds)
    # # print(posenet_preds)
    # if pipeline_cfg.vis:
    #     visualize(image, posenet_preds)
    # return posenet_preds

    posenet_preds_list = []

    frames = get_frames(video)

    start = time.time()
    for i, image in enumerate(frames):
        print(str(i+1) + " / " + str(len(frames)))
        person_boxes = get_image_bounding_boxes(image, detectnet_model)
        print(len(person_boxes))
        if len(person_boxes) == 0:
            continue
        person_images, k_values = get_input(image, person_boxes)
        rootnet_preds = get_root(image, person_boxes, rootnet_model, person_images, k_values)
        posenet_preds = get_pose(image, person_boxes, posenet_model, person_images, rootnet_preds)
        posenet_preds_list.append(posenet_preds)

    print("It takes ", str(time.time() - start) + " s")

    output_file = open(pipeline_cfg.output_video_path, "wb")
    pickle.dump(posenet_preds_list, output_file)
    output_file.close()


def load(binary_path):
    with open(binary_path, "rb") as f:
        data = pickle.load(f)
    return data


def get_header():
    header_list = []
    for i in range(pipeline_cfg.joint_num):
        header_list.extend([str(i)+"_x", str(i)+"_y"])
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


if __name__ == "__main__":
    main()