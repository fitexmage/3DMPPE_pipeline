import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import math

from rootnet_repo.data.dataset import generate_patch_image
from rootnet_repo.main.config import cfg as rootnet_cfg
from rootnet_repo.common.base import Tester as rootnet_Tester
from rootnet_repo.common.utils.pose_utils import pixel2cam

from config import cfg as pipeline_cfg

def get_input(image, person_boxes):
    person_images = np.zeros((len(person_boxes), 3, rootnet_cfg.input_shape[0], rootnet_cfg.input_shape[1]))
    k_values = np.zeros((len(person_boxes), 1))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=rootnet_cfg.pixel_mean, std=rootnet_cfg.pixel_std)]
    )

    for i, box in enumerate(person_boxes):
        image, _ = generate_patch_image(image, box, False, 0)
        image = transform(image)
        person_images[i] = image
        k_values[i] = np.array(
            [math.sqrt(rootnet_cfg.bbox_real[0] * rootnet_cfg.bbox_real[1] * pipeline_cfg.f[0] * pipeline_cfg.f[1] / (box[3] * box[2]))]).astype(
            np.float32)

    person_images = torch.Tensor(person_images)
    k_values = torch.Tensor(k_values)

    return person_images, k_values

def get_root(person_boxes, person_images, k_values):
    rootnet_cfg.set_args(gpu_ids='0')
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    rootnet_tester = rootnet_Tester(pipeline_cfg.rootnet_model_inx)
    rootnet_tester._make_model()

    with torch.no_grad():
        rootnet_preds = rootnet_tester.model(person_images, k_values)
        rootnet_preds = rootnet_preds.cpu().numpy()
    print(rootnet_preds)
    for i, box in enumerate(person_boxes):
        rootnet_pred = rootnet_preds[i]
        rootnet_pred[0] = rootnet_pred[0] / rootnet_cfg.output_shape[1] * box[2] + box[0]
        rootnet_pred[1] = rootnet_pred[1] / rootnet_cfg.output_shape[0] * box[3] + box[1]
        # cv2.circle(im, (rootnet_preds[i][0], rootnet_preds[i][1]), 5, (0, 0, 255), -1)
        if pipeline_cfg.get_3d:
            rootnet_pred[0], rootnet_pred[1], rootnet_pred[2] = pixel2cam(rootnet_pred, pipeline_cfg.f, np.array([box[2]/2, box[3]/2]))
    return rootnet_preds