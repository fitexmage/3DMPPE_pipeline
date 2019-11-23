import numpy as np
import cv2
import math
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from rootnet_repo.main.config import cfg as rootnet_cfg
from rootnet_repo.common.base import Tester as rootnet_Tester
from rootnet_repo.data.dataset import generate_patch_image

from posenet_repo.main.config import cfg as posenet_cfg
from posenet_repo.common.base import Tester as posenet_Test
from posenet_repo.common.utils.pose_utils import warp_coord_to_original

def main():

    im = cv2.imread("./input.jpg")

    detectron_cfg = get_cfg()
    detectron_cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
    detectron_cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    predictor = DefaultPredictor(detectron_cfg)
    outputs = predictor(im)

    # from detectron2.utils.visualizer import Visualizer
    # from detectron2.data import MetadataCatalog
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(detectron_cfg.DATASETS.TRAIN[0]), scale=1.2)
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imwrite("output.jpg", v.get_image()[:, :, ::-1])

    person_boxes = outputs["instances"].pred_boxes[outputs["instances"].pred_classes == 0]

    if len(person_boxes) == 0:
        return

    list = []
    for i, box in enumerate(person_boxes):
        box = box.cpu().numpy()
        box = np.array([box[0], box[1], box[2] - box[0], box[3] - box[1]])
        list.append(box)
    person_boxes = list

    person_images = np.zeros((len(person_boxes), 3, 256, 256))
    k_values = np.zeros((len(person_boxes), 1))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=rootnet_cfg.pixel_mean, std=rootnet_cfg.pixel_std)]
    )

    for i, box in enumerate(person_boxes):
        image, _ = generate_patch_image(im, box, False, 0)
        image = transform(image)
        person_images[i] = image
        k_values[i] = np.array([math.sqrt(rootnet_cfg.bbox_real[0] * rootnet_cfg.bbox_real[1] * 1500 * 1500 / (box[3] * box[2]))]).astype(np.float32)

    person_images = torch.Tensor(person_images)
    k_values = torch.Tensor(k_values)

    rootnet_cfg.set_args(gpu_ids='0')
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    rootnet_tester = rootnet_Tester(18)
    rootnet_tester._make_model()

    with torch.no_grad():
        rootnet_preds = rootnet_tester.model(person_images, k_values)
        rootnet_preds = rootnet_preds.cpu().numpy()

    for i, box in enumerate(person_boxes):
        rootnet_pred = rootnet_preds[i]
        rootnet_pred[0] = rootnet_pred[0] / rootnet_cfg.output_shape[1] * box[2] + box[0]
        rootnet_pred[1] = rootnet_pred[1] / rootnet_cfg.output_shape[0] * box[3] + box[1]
        # cv2.circle(im, (rootnet_preds[i][0], rootnet_preds[i][1]), 5, (0, 0, 255), -1)

    posenet_cfg.set_args('0')

    posenet_tester = posenet_Test(24)
    joint_num = 21
    posenet_tester.joint_num = joint_num
    posenet_tester._make_model()

    with torch.no_grad():
        posenet_preds = posenet_tester.model(person_images)
        posenet_preds = posenet_preds.cpu().numpy()

    for i, box in enumerate(person_boxes):
        posenet_pred = posenet_preds[i]
        posenet_pred[:, 0], posenet_pred[:, 1], posenet_pred[:, 2] = warp_coord_to_original(posenet_pred, box, rootnet_pred[i])

        for joint in posenet_pred:
            cv2.circle(im, (joint[0], joint[1]), 5, (0, 0, 255), 0)
    cv2.imwrite("output.jpg", im)

if __name__ == "__main__":
    main()