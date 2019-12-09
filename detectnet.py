import numpy as np
import cv2
import torch

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def get_detectnet_config():
    detectnet_cfg = get_cfg()
    detectnet_cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    detectnet_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    detectnet_cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    return detectnet_cfg

def get_detectnet_model(detectron_cfg):
    return DefaultPredictor(detectron_cfg)

def get_image_bounding_boxes(image, predictor):
    outputs = predictor(image)

    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    detectnet_config = get_detectnet_config()
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(detectnet_config.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("output.jpg", v.get_image()[:, :, ::-1])

    is_human = outputs["instances"].pred_classes == 0
    high_score = outputs["instances"].scores >= 0.9
    person_boxes = outputs["instances"].pred_boxes[is_human & high_score]

    result = []
    for i, box in enumerate(person_boxes):
        box = box.cpu().numpy()
        # tmp_image = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        # resolution = cv2.Laplacian(cv2.cvtColor(tmp_image, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
        # print(resolution)
        # if resolution < 200:
        #     continue

        ratio = (box[2] - box[0]) * (box[3] - box[1]) / (image.shape[0] * image.shape[1])
        if ratio > 0.01:
            box = np.array([box[0], box[1], box[2] - box[0], box[3] - box[1]])
            result.append(box)

    return result

