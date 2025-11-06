import cv2
import numpy as np
import laserDetector as ld


class CV_Manager:
    def __init__(self):
        self.green_laser = ld.LaserDetector(np.array([[[0, 255, 0]]], dtype=np.uint8))
        self.red_laser = ld.LaserDetector(np.array([[[0, 0, 255]]], dtype=np.uint8))

    def find_distance(self):
        # for testing
        self.green_laser.image = cv2.imread("../test_images/img.png")
        self.red_laser.image = cv2.imread("../test_images/img.png")
        # end of testing code

        distance = 0
        self.green_laser.detect_laser()
        self.red_laser.detect_laser()
        green_pos, green_axis, green_angle = self.green_laser.ellipse
        green_pos = np.array(green_pos)
        red_pos, red_axis, red_angle = self.red_laser.ellipse
        red_pos = np.array(red_pos)
        print(green_pos, green_axis, green_angle)
        print(red_pos, red_axis, red_angle)
        distance = np.linalg.norm(green_pos - red_pos)
        return distance


if __name__ == "__main__":
    cv_manager = CV_Manager()
    print(f"distance calculated: {cv_manager.find_distance()*0.24/5}")