import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import cfg as pipeline_cfg
from posenet_repo.common.utils.vis import vis_keypoints


def visualize_and_save(image, preds):
    if pipeline_cfg.to_camera:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(pipeline_cfg.skeleton) + 2)]
        colors = [np.array((c[2], c[1], c[0])) for c in colors]

        min_z = np.min(preds[:, 0, 2])
        max_z = np.max(preds[:, 0, 2])

        for pred in preds:
            if (pred[0, 0] < -2500) or pred[0, 1] < -2500 or pred[0, 0] > 2500 or pred[0, 1] > 2500 or pred[0, 2] > min_z + 5000:
                continue

            for l in range(len(pipeline_cfg.skeleton)):
                i1 = pipeline_cfg.skeleton[l][0]
                i2 = pipeline_cfg.skeleton[l][1]
                x = np.array([pred[i1, 0], pred[i2, 0]])
                y = np.array([pred[i1, 1], pred[i2, 1]])
                z = np.array([pred[i1, 2], pred[i2, 2]])

                ax.plot(x, z, -y, c=colors[l], linewidth=2)
                ax.scatter(pred[i1, 0], pred[i1, 2], -pred[i1, 1], c=colors[l], marker='o')
                ax.scatter(pred[i2, 0], pred[i2, 2], -pred[i2, 1], c=colors[l], marker='o')

        ax.set_title('3D vis')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Z Label')
        ax.set_zlabel('Y Label')
        ax.set_xlim([-2000,2000])
        ax.set_ylim([min_z - 1000, min_z + 3000])
        ax.set_zlim([-2000,2000])
        ax.legend()
        plt.savefig(pipeline_cfg.output_vis_path)
    else:
        tmpimg = image.copy()
        for pred in preds:
            tmpimg = vis_keypoints(tmpimg, np.transpose(pred), pipeline_cfg.skeleton)

        cv2.imwrite(pipeline_cfg.output_vis_path, tmpimg)
