import numpy as np
import cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def get_bounding_boxes(image):
    detectron_cfg = get_cfg()
    detectron_cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    detectron_cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    predictor = DefaultPredictor(detectron_cfg)
    outputs = predictor(image)

    # from detectron2.utils.visualizer import Visualizer
    # from detectron2.data import MetadataCatalog
    # v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(detectron_cfg.DATASETS.TRAIN[0]), scale=1.2)
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imwrite("output.jpg", v.get_image()[:, :, ::-1])

    person_boxes = outputs["instances"].pred_boxes[outputs["instances"].pred_classes == 0]

    result = []
    for i, box in enumerate(person_boxes):
        box = box.cpu().numpy()
        box = np.array([box[0], box[1], box[2] - box[0], box[3] - box[1]])
        tmp_image = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        if cv2.Laplacian(cv2.cvtColor(tmp_image, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var() < 500:
            continue
        result.append(box)

    return result