import numpy as np
import cv2 as cv

# TODO calibration program?
MARKER_COLOR = np.array((177, 120, 194), dtype='uint8')

def main_subtraction():
    cap = cv.VideoCapture(-1)  # Open the first camera connected to the computer.

    camera_matrix = np.loadtxt("camera_matrix.npy")
    distortion_coeff = np.loadtxt("distortion_coeff.npy")

    ret, frame = cap.read()
    h, w = frame.shape[:2]
    new_camera_matrix, camera_crop_region = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w,h), 1, (w,h))
    mapx, mapy = cv.initUndistortRectifyMap(camera_matrix, distortion_coeff, None, new_camera_matrix, (w,h), cv.CV_16SC2)
    
    while True:
        ret, frame = cap.read()
        frame = cv.undistort(frame, camera_matrix, distortion_coeff, None, new_camera_matrix)  # Undistort the image.

        crop_x, crop_y, crop_w, crop_h = camera_crop_region
        frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        #  print(frame.shape)

        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_hsv -= MARKER_COLOR
        frame_hsv = np.abs(frame_hsv)
        frame_hsv[:,:,0] = np.minimum(frame_hsv[:,:,0], 180 - frame_hsv[:,:,0])

        # rotate hue so we can threshold it
        #  print(frame_threshold.shape)
        #  frame[frame_threshold] = (0, 255, 0)

        #  cv.imshow('Calibration Result', frame)  # Show the undistorted image to the screen.
        #  cv.imshow('Calibration Result', frame_threshold)  Show the undistorted image to the screen.
        #  frame_processed = cv.cvtColor(frame_hsv, cv.COLOR_HSV2BGR)
        #  cv.imshow('Calibration Result', frame_processed)
        cv.imshow('a', frame_hsv)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

MARKER_THRESH_LOW = ((165 + 15) % 180, 128, 50)
MARKER_THRESH_HIGH = ((11 + 15) % 180, 255, 255)

def centroid(contour):
    moments = cv.moments(contour)
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return (cx, cy)

def main():
    cap = cv.VideoCapture(-1)  # Open the first camera connected to the computer.

    camera_matrix = np.loadtxt("camera_matrix.npy")
    distortion_coeff = np.loadtxt("distortion_coeff.npy")

    ret, frame = cap.read()
    h, w = frame.shape[:2]
    new_camera_matrix, camera_crop_region = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w,h), 1, (w,h))
    mapx, mapy = cv.initUndistortRectifyMap(camera_matrix, distortion_coeff, None, new_camera_matrix, (w,h), cv.CV_16SC2)

    morphology_kernel = np.ones((5, 5), np.uint8)
    panel_w, panel_h = 1650, 490
    
    while True:
        ret, frame = cap.read()
        frame = cv.undistort(frame, camera_matrix, distortion_coeff, None, new_camera_matrix)  # Undistort the image.

        crop_x, crop_y, crop_w, crop_h = camera_crop_region
        frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        #  print(frame.shape)

        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_hsv[:,:,0] += 15
        frame_hsv[:,:,0] %= 180
        # Mask any pixels that are within range of colored squares
        thresh_img = cv.inRange(frame_hsv, MARKER_THRESH_LOW, MARKER_THRESH_HIGH)

        # Blur and morph open to remove noise
        thresh_img = cv.blur(thresh_img, (5, 5))
        # NOTE for report: the dilation fucks it up
        #  thresh_img = cv.morphologyEx(thresh_img, cv.MORPH_OPEN, morphology_kernel, iterations=2)
        #  thresh_img = cv.morphologyEx(thresh_img, cv.MORPH_DILATE, morphology_kernel, iterations=5)

        contour_img = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3))
        contours, hierarchy = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(contour_img, contours, -1, (255, 0, 0), 1)

        if len(contours) == 4:
            # There should be for contours (ie. the outline of the four colored squares)
            contours_concat = np.vstack((contours[i] for i in range(len(contours))))
            # Create a single convex contour from the four
            contours_hull = cv.convexHull(contours_concat)

            # Simplify into (hopefully) a quadrilateral
            epsilon = 0.05 * cv.arcLength(contours_hull, True)
            hull_quad = cv.approxPolyDP(contours_hull, epsilon, True)
            if len(hull_quad) == 4:
                hull_points = hull_quad.reshape((4, 2))

                # Find the point that is closest to the top left (0, 0)
                topleft_index = min(range(len(hull_points)), key=lambda x: hull_points[x, 0]**2 + hull_points[x, 1]**2)
                # left shift the array such that the top-left point is at index 0
                hull_points = np.roll(hull_points, -topleft_index, axis=0)
                cv.circle(contour_img, tuple(hull_points[0]), 10, (255, 0, 0))
                #  cv.circle(contour_img, tuple(hull_points[2]), 10, (0, 0, 255))
                cv.drawContours(contour_img, [contours_hull], -1, (0, 255, 0), 1)
                cv.drawContours(contour_img, [hull_quad], -1, (0, 0, 255), 1)
                
                hull_points = np.float32(hull_points)
                
                transform = cv.getPerspectiveTransform(hull_points, np.float32([[0, 0], [panel_w, 0], [panel_w, panel_h], [0, panel_h]]))
                warped = cv.warpPerspective(frame, transform, (panel_w, panel_h))
                cv.imshow('b', warped)




        #  cv.imshow('Calibration Result', frame)  # Show the undistorted image to the screen.
        #  cv.imshow('Calibration Result', frame_threshold)  Show the undistorted image to the screen.
        #  frame_processed = cv.cvtColor(frame_hsv, cv.COLOR_HSV2BGR)
        #  cv.imshow('Calibration Result', frame_processed)
        cv.imshow('a', contour_img)
        #  cv.imshow('b', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
main()
