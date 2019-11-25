import torch
import numpy as np

from posenet_repo.main.config import cfg as posenet_cfg
from posenet_repo.common.base import Tester as posenet_Test
from posenet_repo.common.utils.pose_utils import pixel2cam, warp_coord_to_original, flip

from config import cfg as pipeline_cfg

def get_pose(raw_image, person_boxes, person_images, rootnet_preds):
    posenet_cfg.set_args('0')

    posenet_tester = posenet_Test(pipeline_cfg.posenet_model_inx)
    posenet_tester.joint_num = pipeline_cfg.joint_num
    posenet_tester._make_model()

    with torch.no_grad():
        posenet_preds = posenet_tester.model(person_images)
        if pipeline_cfg.flip_test:
            flipped_input_img = flip(person_images, dims=3)
            flipped_coord_out = posenet_tester.model(flipped_input_img)
            flipped_coord_out[:, :, 0] = posenet_cfg.output_shape[1] - flipped_coord_out[:, :, 0] - 1

            for pair in pipeline_cfg.flip_pairs:
                flipped_coord_out[:, pair[0], :], flipped_coord_out[:, pair[1], :] = flipped_coord_out[:, pair[1], :].clone(), flipped_coord_out[:, pair[0], :].clone()

            posenet_preds = (posenet_preds + flipped_coord_out) / 2.
        posenet_preds = posenet_preds.cpu().numpy()

    for i, box in enumerate(person_boxes):
        posenet_pred = posenet_preds[i]
        posenet_pred[:, 0], posenet_pred[:, 1], posenet_pred[:, 2] = warp_coord_to_original(posenet_pred, box, rootnet_preds[i])

        if pipeline_cfg.to_camera:
            posenet_pred[:, 0], posenet_pred[:, 1], posenet_pred[:, 2] = pixel2cam(posenet_pred, pipeline_cfg.f, np.array([raw_image.shape[1]/2, raw_image.shape[0]/2]))

    return posenet_preds