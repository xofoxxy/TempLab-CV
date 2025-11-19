import cv2
import numpy as np
import laserDetector as ld


class CV_Manager:
    def __init__(self):
        self.green = np.array([[[0, 255, 0]]], dtype=np.uint8)
        self.red = np.array([[[0, 0, 255]]], dtype=np.uint8)
        self.green_laser = ld.LaserDetector(self.green)
        self.red_laser = ld.LaserDetector(self.red)

    def find_distance(self):
        # Here we collect the image from the buffer

        # for testing
        self.image = cv2.imread("../test_images/img.png")
        self.green_laser.image = self.image.copy()
        self.red_laser.image = self.image.copy()
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

    def characterize_lasers(self):
        def gradient_along_axis(ellipse, color, minor_axis):
            THRESHOLD = .7
            pixels_of_interest = []
            pos, axis, angle = ellipse
            if minor_axis is True:
                angle += 90
            print(f"Angle of {'minor axis' if minor_axis else 'major axis'} gradient: {angle} Degrees")
            color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            image = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2HSV)
            y, x = int(round(pos[1])), int(round(pos[0]))
            pixel = image[y, x]
            gradient_list = []
            gradient_list.append(((y,x), pixel))
            i = 0
            while (1 - THRESHOLD) * 255 <= pixel[2] <= 255 and THRESHOLD * color[0][0][0] <= pixel[0] <= (2 - THRESHOLD) * \
                    color[0][0][0]:  # this will check the saturation and value
                # walk in the positive direction and collect data
                x_change = np.cos(angle) * i
                y_change = np.sin(angle) * i # we need to round because cos and sin are floats and we need pixel values
                new_x = x + x_change
                new_y = y + y_change
                i += 1
                if (new_y, new_x) == (y, x):
                    # if it rounded to the same position, we need to do it again
                    continue
                else:
                    print(f"iterations required: {i}")
                    i = 0
                    y, x = new_y, new_x
                    pixel_y, pixel_x = int(round(y)), int(round(x))
                    pixel = image[pixel_y, pixel_x]
                    gradient_list.append(((pixel_y,pixel_x), pixel))
            # now we need to walk in the negative direction doing the exact same thing
            i = 0
            y, x = int(round(pos[1])), int(round(pos[0]))# reset the position to the starting point
            pixel = image[y, x]
            print("Now going backwards...")
            while (1 - THRESHOLD) * 255 <= pixel[2] <= 255 and THRESHOLD * color[0][0][0] <= pixel[0] <= (2 - THRESHOLD) * \
                    color[0][0][0]:  # this will check the saturation and value
                # walk in the negative direction and collect data
                x_change = np.cos(angle) * -i
                y_change = np.sin(angle) * -i # we need to round because cos and sin are floats and we need pixel values
                new_x = x + x_change
                new_y = y + y_change
                i += 1
                if (new_y, new_x) == (y, x):
                    # if it rounded to the same position, we need to do it again
                    continue
                else:
                    print(f"iterations required: {i}")
                    i = 0
                    y, x = new_y, new_x
                    pixel_y, pixel_x = int(round(y)), int(round(x))
                    pixel = image[pixel_y, pixel_x]
                    gradient_list.append(((pixel_y, pixel_x), pixel))

            return gradient_list

        green_pos, green_axis, green_angle = self.green_laser.ellipse
        red_pos, red_axis, red_angle = self.red_laser.ellipse
        green_pos = np.array(green_pos)
        red_pos = np.array(red_pos)

        green_along_major = gradient_along_axis(self.green_laser.ellipse, self.green, False)
        green_along_minor = gradient_along_axis(self.green_laser.ellipse, self.green, True)
        print(green_along_major)
        print(green_along_minor)

        print(f"Number of pixels along major axis: {len(green_along_major)}")
        print(f"Number of pixels along minor axis: {len(green_along_minor)}")




if __name__ == "__main__":
    cv_manager = CV_Manager()
    print(f"distance calculated: {cv_manager.find_distance() * 0.24 / 5}")
    cv_manager.characterize_lasers()
