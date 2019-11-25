import numpy as np


class Config:
    joint_num = 21
    joints_name = ('Head_top',  # 0
                   'Thorax',  # 1
                   'R_Shoulder',  # 2
                   'R_Elbow',  # 3
                   'R_Wrist',  # 4
                   'L_Shoulder',  # 5
                   'L_Elbow',  # 6
                   'L_Wrist',  # 7
                   'R_Hip',  # 8
                   'R_Knee',  # 9
                   'R_Ankle',  # 10
                   'L_Hip',  # 11
                   'L_Knee',  # 12
                   'L_Ankle',  # 13
                   'Pelvis',  # 14
                   'Spine',  # 15
                   'Head',  # 16
                   'R_Hand',  # 17
                   'L_Hand',  # 18
                   'R_Toe',  # 19
                   'L_Toe'  # 20
                   )

    flip_test = False
    flip_pairs = ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13))
    rootnet_model_inx = 18
    posenet_model_inx = 24
    f = np.array([1500, 1500])

    get_3d = True


cfg = Config()
