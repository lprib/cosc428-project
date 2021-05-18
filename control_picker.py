import numpy as np
import cv2 as cv
import sys
from util import run_camera_loop

def main():
    img = cv.imread("data/transformed3.png", -1)
    boxes = []
    current_start = (0, 0)

    m = (0, 0)
    def mouse_callback(event, x, y, flags, param):
        nonlocal m
        m = (x, y)

    cv.namedWindow("main")
    cv.setMouseCallback("main", mouse_callback)

    while True:
        drawing = img.copy()
        for box in boxes:
            cv.rectangle(drawing, box[0], box[1], (255, 0, 0), 1)

        cv.rectangle(drawing, current_start, m, (0, 0, 255), 2)
        cv.circle(drawing, ((m[0] + current_start[0])//2, (m[1] + current_start[1])//2), 5, (255, 255, 255), 1)

        key = cv.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        if key == ord('a'):
            current_start = m
        if key == ord('z'):
            boxes.append((current_start, m))

        cv.imshow("main", drawing)

    filename = sys.argv[1] if len(sys.argv) > 1 else "data/knobs.csv"
    with open(filename, "w") as csv_file:
        csv_file.write("x1,y1,x2,y2\n")
        for box in boxes:
            csv_file.write(f"{box[0][0]},{box[0][1]},{box[1][0]},{box[1][1]}\n")

main()
