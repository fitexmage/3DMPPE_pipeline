import numpy as np


class Config:
    input_inx = 1
    input_path = "data/input_" + str(input_inx) + ".jpg"
    output_path = "data/output_" + str(input_inx) + ".jpg"

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
    skeleton = ((0, 16),
                (16, 1),
                (1, 15),
                (15, 14),
                (14, 8),
                (14, 11),
                (8, 9),
                (9, 10),
                # (10, 19),
                (11, 12),
                (12, 13),
                # (13, 20),
                (1, 2),
                (2, 3),
                (3, 4),
                # (4, 17),
                (1, 5),
                (5, 6),
                (6, 7),
                # (7, 18)
                )
    flip_test = False
    flip_pairs = ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13))
    rootnet_model_inx = 18
    posenet_model_inx = 24
    # f = np.array([800, 800])

    to_camera = True
    vis = True

cfg = Config()
