# import some common libraries
import numpy as np
import cv2
import random

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
        box = box.cpu().numpy().astype(int)
        image = im[box[1]:box[3], box[0]: box[2]]
        person_images.append(image)

    import argparse
    from rootnet_repo.main.config import cfg
    import torch
    from rootnet_repo.common.base import Tester
    from rootnet_repo.common.utils.pose_utils import flip
    import torch.backends.cudnn as cudnn
    import math
    import torchvision.transforms as transforms

    cfg.set_args(gpu_ids='0')
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    tester = Tester(18)
    tester._make_model()

    preds = []

    for image in person_images:
        k_value = np.array([math.sqrt(2000 * 2000 * 30 * 30 / (image.shape[0] * image.shape[1]))]).astype(np.float32)
        k_value = torch.Tensor(k_value)

        image = cv2.resize(image, (256, 256))
        transform = transforms.Compose([ \
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)] \
            )
        image = transform(image)
        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            coord_out = tester.model(image, k_value)
            coord_out = coord_out.cpu().numpy()
            preds.append(coord_out)
    preds = np.concatenate(preds, axis=0)
    print(preds)

if __name__ == "__main__":
    main()