import numpy as np
import cv2
import math
import torch
import torchvision.transforms as transforms

from rootnet_repo.data.dataset import generate_patch_image

from detectnet import get_bounding_boxes
from rootnet import get_root
from posenet import get_pose
from config import cfg as pipeline_cfg

def get_input(image, person_boxes):
    person_images = np.zeros((len(person_boxes), 3, pipeline_cfg.input_shape[0], pipeline_cfg.input_shape[1]))
    k_values = np.zeros((len(person_boxes), 1))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=pipeline_cfg.pixel_mean, std=pipeline_cfg.pixel_std)]
    )

    for i, box in enumerate(person_boxes):
        image, _ = generate_patch_image(image, box, False, 0)
        image = transform(image)
        person_images[i] = image
        k_values[i] = np.array(
            [math.sqrt(pipeline_cfg.bbox_real[0] * pipeline_cfg.bbox_real[1] * 1500 * 1500 / (box[3] * box[2]))]).astype(
            np.float32)

    person_images = torch.Tensor(person_images)
    k_values = torch.Tensor(k_values)

    return person_images, k_values

def main():
    image = cv2.imread("./input.jpg")
    person_boxes = get_bounding_boxes(image)
    if len(person_boxes) == 0:
        return

    person_images, k_values = get_input(image, person_boxes)
    rootnet_preds = get_root(person_boxes, person_images, k_values)
    posenet_preds = get_pose(person_boxes, person_images, rootnet_preds)
    # cv2.imwrite("output.jpg", image)

if __name__ == "__main__":
    main()