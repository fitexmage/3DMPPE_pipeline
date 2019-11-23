
from rootnet_repo.main.config import cfg as rootnet_cfg

class Config:
    joint_num = 21
    joints_name = ('Head_top',
                   'Thorax',
                   'R_Shoulder',
                   'R_Elbow',
                   'R_Wrist',
                   'L_Shoulder',
                   'L_Elbow',
                   'L_Wrist',
                   'R_Hip',
                   'R_Knee',
                   'R_Ankle',
                   'L_Hip',
                   'L_Knee',
                   'L_Ankle',
                   'Pelvis',
                   'Spine',
                   'Head',
                   'R_Hand',
                   'L_Hand',
                   'R_Toe',
                   'L_Toe')
    flip_test = False
    flip_pairs = ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13))
    rootnet_model_inx = 18
    posenet_model_inx = 24

    input_shape = rootnet_cfg.input_shape
    output_shape = rootnet_cfg.output_shape
    pixel_mean = rootnet_cfg.pixel_mean
    pixel_std = rootnet_cfg.pixel_std
    bbox_real = rootnet_cfg.bbox_real

cfg = Config()