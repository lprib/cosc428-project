import numpy as np
import cv2 as cv
import csv
from transform_color_mark import transform

def setup_camera(cam_w, cam_h, camera_matrix_file, distortion_coeff_file):
    cap = cv.VideoCapture(-1)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, cam_w)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cam_h)

    camera_matrix = np.loadtxt(camera_matrix_file)
    distortion_coeff = np.loadtxt(distortion_coeff_file)

    ret, frame = cap.read()
    h, w = frame.shape[:2]
    new_camera_matrix, camera_crop_region = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w,h), 1, (w,h))
    mapx, mapy = cv.initUndistortRectifyMap(camera_matrix, distortion_coeff, None, new_camera_matrix, (w,h), cv.CV_16SC2)

    # return in the args format expected by cv.undistort
    return cap, camera_crop_region, (camera_matrix, distortion_coeff, None, new_camera_matrix)


def preprocess_frame(setup_camera_info):
    cap, crop, distort_info = setup_camera_info
    ret, frame = cap.read()
    frame = cv.undistort(frame, *distort_info)

    crop_x, crop_y, crop_w, crop_h = crop
    return frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]


def run_camera_loop(cam_w, cam_h, camera_matrix_file, distortion_coeff_file, mouse_window_name, callback):
    cam_info = setup_camera(cam_w, cam_h, camera_matrix_file, distortion_coeff_file)

    mouse_x, mouse_y = 0, 0
    def mouse_callback(event, x, y, flags, param):
        # Closure capture:
        nonlocal mouse_x
        nonlocal mouse_y
        mouse_x, mouse_y = x, y

    cv.namedWindow(mouse_window_name)
    cv.setMouseCallback(mouse_window_name, mouse_callback)

    while True:
        frame = preprocess_frame(cam_info)

        key = cv.waitKey(20) & 0xFF
        if key == ord('q'):
            break

        callback(frame, key, mouse_x, mouse_y)

def get_control_positions():
    controls = []
    with open("data/knobs.csv", newline="") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        _header = next(reader)
        for row in reader:
            controls.append(tuple(int(x) for x in row))
    return controls

def sub_image(img, rect):
    return img[rect[1]:rect[3], rect[0]:rect[2]].copy()

def pad_to_width(img, width):
    return cv.copyMakeBorder(img, 0, 0, 0, width - img.shape[1], cv.BORDER_CONSTANT, value=(0, 0, 0))

def control_detect_test_static(input_image_name, *args, **kwargs):
    main_img = cv.imread(input_image_name, -1)
    control_detect_test(lambda: main_img, *args, **kwargs)

def control_detect_test_video(*args, **kwargs):
    cam_info = setup_camera(1280, 720, "data/camera_matrix_720.npy", "data/distortion_coeff_720.npy")

    blank = np.zeros((490, 1650, 3), dtype=np.uint8)
    def image_producer():
        nonlocal blank, cam_info
        frame = preprocess_frame(cam_info)
        success, transformed = transform(frame, draw_debug=True, debug_scale=0.5)
        return (transformed if success else blank), frame

    control_detect_test(image_producer, *args, **kwargs)


def control_detect_test(image_producer, test_function, win_name, trackbar_info, control_indices=None, draw_ref_img=False):
    """
    win_name: name of window
    trackbar_info: iterator of (trackbar_name, trackbar_value, trackbar_max_value)
    input_image_name: filename of input (controls will be extracted)
    test_function: takes (sub_image, trackbar_data0, trackbar_data1, ...) and returns an array of images
        NOTE: the images should be equal width and height
    """
    cv.namedWindow(win_name)

    for name, val, maxval in trackbar_info:
        cv.createTrackbar(name, win_name, val, maxval, lambda x: None)
    
    controls = get_control_positions()
    if control_indices is not None:
        controls = [c for i, c in enumerate(controls) if i in control_indices]


    while True:
        main_img, cam_img = image_producer()
        if draw_ref_img:
            cv.imshow("reference_image", main_img)
            cv.imshow("camera_image", cam_img)

        trackbar_data = tuple(cv.getTrackbarPos(name, win_name) for name, _, _ in trackbar_info)

        keys = cv.waitKey(20) & 0xff
        if keys == ord('q'):
            cv.destroyAllWindows()
            break

        # Get the test output for each controller test input
        # This will be an array of (array of image)
        test_outputs = [test_function(sub_image(main_img, control_box), keys, *trackbar_data) for control_box in controls]

        # get the max width of a column, for padding
        max_widths = [
            max(test_outputs, key=lambda x: x[col_index].shape[1])[col_index].shape[1]
            for col_index in range(len(test_outputs[0]))
        ]

        # Pad columns and concatenate rows
        test_outputs_padded = [
            np.concatenate([pad_to_width(img, max_widths[col_index]) for col_index, img in enumerate(test_output)], axis=1)
            for test_output in test_outputs
        ]

        combined = np.concatenate(test_outputs_padded, axis=0)

        #  print(lines.shape)
        cv.imshow(win_name, combined)

