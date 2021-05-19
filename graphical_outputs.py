import sys
import cv2 as cv
import numpy as np
from transform_color_mark import transform
from util import get_control_positions, sub_image
from control_detector_hough_lines import do_hough_image


cam = cv.imread("data/camera.png", -1)
succ, transformed, contour, thresh = transform(cam, draw_debug=True)
if not succ:
    print("couldn't transform")
    sys.exit(1)

cv.imwrite("outputs/camera.png", cam)
cv.imwrite("outputs/contour.png", contour)
ret, thresh = cv.threshold(thresh, 10, 255, cv.THRESH_BINARY)
cv.imwrite("outputs/thresh.png", thresh)
cv.imwrite("outputs/transformed.png", transformed)

control = get_control_positions()[1]
control_img = sub_image(transformed, control)

images = do_hough_image(control_img, -1, 49, 144, 10, 10)
images = np.concatenate(images, axis=0)
cv.imwrite("outputs/hough.png", images)
