import detectron2

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

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
        box = box.cpu().numpy()
        print(box)
        image = im[box[0]:box[2], box[1]: box[3]]
        person_images.append(image)
    cv2.imwrite("output.jpg", person_images[0])

if __name__ == "__main__":
    main()