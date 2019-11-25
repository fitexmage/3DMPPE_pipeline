from config import cfg as pipeline_cfg
from rootnet_repo.common.utils.vis import vis_keypoints, vis_3d_skeleton

def vis(image, keypoints):
    if pipeline_cfg.get_3d:
        vis_3d_skeleton()
    else:
        vis_image = vis_keypoints(image, keypoints, pipeline_cfg.skeleton)