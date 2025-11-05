import os

import numpy as np

from src.laserDetector import LaserDetector
import cv2

if __name__ == "__main__":
    test_image = cv2.imread("test_images/img.png")
    images = [test_image]
    color = np.array([[[0, 255, 0]]], dtype=np.uint8)
    laserDetector = LaserDetector(color, image=test_image)
    mask = laserDetector.mask(color)
    masked = cv2.bitwise_and(test_image, test_image, mask=mask)
    cv2.imshow("mask", masked)
    cv2.waitKey(0)
    images.append(masked)
    contours = laserDetector.find_contours()
    cv2.imshow("contours", contours)
    cv2.waitKey(0)
    images.append(contours)

    # Resize images to a common height
    # max_height = max(img.shape[0] for img in images)
    # resized_images = [cv2.resize(img, (int(img.shape[1] * max_height / img.shape[0]), max_height)) for img in images]
    #
    # # Concatenate images horizontally
    # combined_image = np.hstack(resized_images)

    # cv2.imshow("steps", combined_image)
    # cv2.waitKey(0)