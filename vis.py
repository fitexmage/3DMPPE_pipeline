import cv2
import numpy as np

from config import cfg as pipeline_cfg
from posenet_repo.common.utils.vis import vis_keypoints, vis_3d_skeleton
from posenet_repo.main.config import cfg as posenet_cfg

def vis(image, preds):
    tmpimg = image.cpu().numpy()
    tmpimg = tmpimg * np.array(posenet_cfg.pixel_std).reshape(3, 1, 1) + np.array(posenet_cfg.pixel_mean).reshape(3, 1, 1)
    tmpimg = tmpimg.astype(np.uint8)
    tmpimg = tmpimg[::-1, :, :]
    tmpimg = np.transpose(tmpimg, (1, 2, 0)).copy()
    tmpkps = np.zeros((3, pipeline_cfg.joint_num))
    tmpkps[:2, :] = preds[0, :, :2].cpu().numpy().transpose(1, 0) / pipeline_cfg.output_shape[0] * pipeline_cfg.input_shape[0]
    tmpkps[2, :] = 1
    tmpimg = vis_keypoints(tmpimg, tmpkps, pipeline_cfg.skeleton)
    cv2.imwrite('output.jpg', tmpimg)