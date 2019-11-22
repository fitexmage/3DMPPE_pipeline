# import some common libraries
import numpy as np
import cv2
import random

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import argparse
from tqdm import tqdm
from rootnet_repo.main.config import cfg
import torch
from rootnet_repo.common.base import Tester
from rootnet_repo.common.utils.vis import vis_keypoints
from rootnet_repo.common.utils.pose_utils import flip
import torch.backends.cudnn as cudnn

def main():
    im = cv2.imread("./input.jpg")
    cfg = get_cfg()
    cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    person_boxes = outputs["instances"].pred_boxes[outputs["instances"].pred_classes == 0]
    person_images = []
    for box in person_boxes:
        box = box.cpu().numpy().astype(int)
        image = im[box[1]:box[3], box[0]: box[2]]
        person_images.append(image)

    test_img = person_images[0]

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    cfg.set_args(0)
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    tester = Tester(18)
    tester._make_batch_generator()
    tester._make_model()

    preds = []

    with torch.no_grad():
        coord_out = tester.model(test_img)
        print(type(coord_out))


if __name__ == "__main__":
    main()