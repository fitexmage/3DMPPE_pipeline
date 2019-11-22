# import some common libraries
import numpy as np
import cv2
import random

def main():
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog

    im = cv2.imread("./input.jpg")

    cfg = get_cfg()
    cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    import argparse
    from rootnet_repo.main.config import cfg
    import torch
    from rootnet_repo.common.base import Tester
    from rootnet_repo.common.utils.pose_utils import flip
    import torch.backends.cudnn as cudnn
    import math
    import torchvision.transforms as transforms
    from rootnet_repo.data.dataset import generate_patch_image

    person_boxes = outputs["instances"].pred_boxes[outputs["instances"].pred_classes == 0]

    person_images = np.zeros((len(person_boxes), 3, 256, 256))
    k_values = np.zeros((len(person_boxes), 1))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]
    )

    i = 0

    for box in person_boxes:
        box = box.cpu().numpy().astype(int)
        image, _ = generate_patch_image(im, box, False, 0)
        for j in range(image.shape[2]):
            image[:, :, j] = np.clip(image[:, :, j], 0, 255)
        if i == 0:
            cv2.imwrite("output.jpg", image)
        image = transform(image)
        person_images[i, :, :image.shape[1], :image.shape[2]] = image
        k_values[i] = np.array([math.sqrt(2000 * 2000 * 35 * 35 / ((box[3] - box[1]) * (box[2] - box[0])))]).astype(np.float32)

        i += 1

    person_images = torch.Tensor(person_images)
    k_values = torch.Tensor(k_values)

    cfg.set_args(gpu_ids='0')
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    tester = Tester(18)
    tester._make_model()

    preds = []

    with torch.no_grad():
        coord_out = tester.model(person_images, k_values)
        coord_out = coord_out.cpu().numpy()
        preds.append(coord_out)
    preds = np.concatenate(preds, axis=0)
    print(preds)

if __name__ == "__main__":
    main()