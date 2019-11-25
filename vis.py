import cv2
import numpy as np

from config import cfg as pipeline_cfg
from posenet_repo.common.utils.vis import vis_keypoints, vis_3d_skeleton
from posenet_repo.main.config import cfg as posenet_cfg

def visualize(image, preds):
    if pipeline_cfg.to_camera:
        print()
    else:
        tmpimg = image.clone()
        for pred in preds:
            tmpimg = vis_keypoints(tmpimg, np.transpose(pred), pipeline_cfg.skeleton)

    cv2.imwrite('output.jpg', tmpimg)

