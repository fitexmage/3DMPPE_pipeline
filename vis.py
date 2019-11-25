import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import cfg as pipeline_cfg
from posenet_repo.common.utils.vis import vis_keypoints
from posenet_repo.main.config import cfg as posenet_cfg


def visualize(image, preds):
    tmpimg = image.copy()
    if pipeline_cfg.to_camera:
        pred = preds[0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(pipeline_cfg.skeleton) + 2)]
        colors = [np.array((c[2], c[1], c[0])) for c in colors]

        for l in range(len(pipeline_cfg.skeleton)):
            i1 = pipeline_cfg.skeleton[l][0]
            i2 = pipeline_cfg.skeleton[l][1]
            x = np.array([pred[i1, 0], pred[i2, 0]])
            y = np.array([pred[i1, 1], pred[i2, 1]])
            z = np.array([pred[i1, 2], pred[i2, 2]])

            if pred[i1, 0] > 0 and pred[i2, 0] > 0:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if pred[i1, 0] > 0:
                ax.scatter(pred[i1, 0], pred[i1, 2], -pred[i1, 1], c=colors[l], marker='o')
            if pred[i2, 0] > 0:
                ax.scatter(pred[i2, 0], pred[i2, 2], -pred[i2, 1], c=colors[l], marker='o')

        x_r = np.array([0, posenet_cfg.input_shape[1]], dtype=np.float32)
        y_r = np.array([0, posenet_cfg.input_shape[0]], dtype=np.float32)
        z_r = np.array([0, 1], dtype=np.float32)

        ax.set_title('3D vis')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Z Label')
        ax.set_zlabel('Y Label')
        ax.legend()
        plt.savefig("output.jpg")
    else:
        for pred in preds:
            tmpimg = vis_keypoints(tmpimg, np.transpose(pred), pipeline_cfg.skeleton)

    cv2.imwrite('output.jpg', tmpimg)

