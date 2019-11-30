import cv2

from detectnet import get_config, get_image_bounding_boxes, get_video_bounding_boxes
from rootnet import get_input, get_root
from posenet import get_pose
from config import cfg as pipeline_cfg
from vis import visualize

def main():
    detectron_config = get_config()
    video = cv2.VideoCapture(pipeline_cfg.input_video_path)
    video_bounding_boxes = get_video_bounding_boxes(video, detectron_config)

    # image = cv2.imread(pipeline_cfg.input_image_path)
    # from detectron2.engine import DefaultPredictor
    # person_boxes = get_image_bounding_boxes(image, DefaultPredictor(detectron_config))
    # if len(person_boxes) == 0:
    #     return

    # person_images, k_values = get_input(image, person_boxes)
    # rootnet_preds = get_root(image, person_boxes, person_images, k_values)
    # # print(rootnet_preds)
    # posenet_preds = get_pose(image, person_boxes, person_images, rootnet_preds)
    # # print(posenet_preds)
    # if pipeline_cfg.vis:
    #     visualize(image, posenet_preds)
    # return posenet_preds

    posenet_preds_list = []

    for i, (image, person_boxes) in enumerate(video_bounding_boxes):
        print(str(i) + " / " + str(len(video_bounding_boxes)))
        if len(person_boxes) == 0:
            continue
        person_images, k_values = get_input(image, person_boxes)
        rootnet_preds = get_root(image, person_boxes, person_images, k_values)
        # print(rootnet_preds)
        posenet_preds = get_pose(image, person_boxes, person_images, rootnet_preds)
        # print(posenet_preds)
        posenet_preds_list.append(posenet_preds)

    print(len(posenet_preds_list))

if __name__ == "__main__":
    main()