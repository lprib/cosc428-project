import cv2 as cv
import numpy as np

DELIM_1 = 1 * np.pi / 4 - 0.4
DELIM_4 = 7 * np.pi / 4 + 0.4
DELIM_2 = DELIM_4 - np.pi
DELIM_3 = DELIM_1 + np.pi


# Returns True if right handed, False if left handed
def do_lr_detection(edges):
    center_x = edges.shape[1] // 2
    left = np.sum(edges[:, :center_x])
    right = np.sum(edges[:, center_x:])

    return left > right


# assumes angle between 0 and pi
def do_angle_correction(angle, left_handed):
    flipped = False

    if angle <= DELIM_1 or angle >= DELIM_4:
        # always flip if in bottom quadrant
        angle += np.pi
        flipped = True
    elif angle >= DELIM_1 and angle <= DELIM_2 and left_handed:
        # if right handed but angle is in left quadrant
        angle += np.pi
        flipped = True
    elif angle >= DELIM_3 and angle <= DELIM_4 and not left_handed:
        # if left handed but angle is in right quadrant
        angle += np.pi
        flipped = True

    return angle % (2 * np.pi), flipped


def draw_angle(drawing, angle, flipped):
    center = (int(drawing.shape[0] / 2), int(drawing.shape[1] / 2))
    cv.circle(drawing, center, 5, (255, 255, 255), 1)
    r = center[0]
    dims = (int(r*np.sin(angle)), int(r*np.cos(angle)))
    color = (0, 0, 255) if flipped else (0, 255, 0)
    cv.line(drawing, center, (center[0] +
            dims[0], center[1] + dims[1]), color, 2)
