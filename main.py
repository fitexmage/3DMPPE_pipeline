import cv2

from detectnet import get_bounding_boxes
from rootnet import get_input, get_root
from posenet import get_pose

def main():
    image = cv2.imread("./input.jpg")
    person_boxes = get_bounding_boxes(image)
    if len(person_boxes) == 0:
        return

    person_images, k_values = get_input(image, person_boxes)
    rootnet_preds = get_root(person_boxes, person_images, k_values)
    print(rootnet_preds.shape)
    posenet_preds = get_pose(person_boxes, person_images, rootnet_preds)
    print(posenet_preds.shape)
    return posenet_preds

if __name__ == "__main__":
    main()