import numpy as np
import cv2 as cv
import csv
from transform_color_mark import transform


def gray(img):
    """helper function to convert img to grayscale"""
    return cv.cvtColor(img, cv.COLOR_GRAY2BGR)


def distance_to_point(start, end, point):
    """
    return minimum distance between point defined by start=(x, y) and end=(x, y)
    and a point defined by (x, y)
    """
    return np.abs(
        (end[0] - start[0])*(start[1] - point[1]) -
        (start[0] - point[0])*(end[1] - start[1])
    ) / np.sqrt(
        (end[0] - start[0])*(end[0] - start[0]) +
        (end[1] - start[1])*(end[1] - start[1])
    )


def setup_camera(cam_w, cam_h, camera_matrix_file, distortion_coeff_file):
    """ Initialize a camera with specified w, h using matrix and distortion files"""
    cap = cv.VideoCapture(-1)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, cam_w)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cam_h)

    camera_matrix = np.loadtxt(camera_matrix_file)
    distortion_coeff = np.loadtxt(distortion_coeff_file)

    ret, frame = cap.read()
    h, w = frame.shape[:2]
    new_camera_matrix, camera_crop_region = cv.getOptimalNewCameraMatrix(
        camera_matrix, distortion_coeff, (w, h), 1, (w, h))
    mapx, mapy = cv.initUndistortRectifyMap(
        camera_matrix, distortion_coeff, None, new_camera_matrix, (w, h), cv.CV_16SC2)

    # return in the args format expected by cv.undistort
    return cap, camera_crop_region, (camera_matrix, distortion_coeff, None, new_camera_matrix)


def preprocess_frame(setup_camera_info):
    """ undistort and crop an image from the camera """
    cap, crop, distort_info = setup_camera_info
    ret, frame = cap.read()
    frame = cv.undistort(frame, *distort_info)

    crop_x, crop_y, crop_w, crop_h = crop
    return frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]


def run_camera_loop(cam_w, cam_h, camera_matrix_file, distortion_coeff_file, mouse_window_name, callback):
    """ Helper function to run a loop that produces a webcam frame each iteration. It will be given to callback """
    cam_info = setup_camera(
        cam_w, cam_h, camera_matrix_file, distortion_coeff_file)

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


def get_control_positions(csv_filename):
    """ returns list of (x1, y1, x2, y2) sum-image bounding boxes """
    controls = []
    with open(csv_filename, newline="") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        _header = next(reader)
        for row in reader:
            controls.append(tuple(int(x) for x in row))
    return controls


def draw_resized(img, name, scale):
    """ show img resized by scale """
    resized_img = cv.resize(
        img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
    cv.imshow(name, resized_img)


def sub_image(img, rect):
    """ rect: (x1, y1, x2, y2) """
    return img[rect[1]:rect[3], rect[0]:rect[2]].copy()


def pad_to_width(img, width):
    """ pad img such that it is width pixels wide """
    return cv.copyMakeBorder(img, 0, 0, 0, width - img.shape[1], cv.BORDER_CONSTANT, value=(0, 0, 0))


def control_detect_test(
    image_producer,
    test_function,
    win_name,
    trackbar_info,
    control_indices=None,
    draw_ref_img=False,
    write_to_video=False,
    draw_trackbars=True
):
    """
    image_producer: callback called once per frame do produce an image. must produce a tuple of (main_image, camera_image)
        main_image will be used to get control sub-images from
        camera_image will be drawn to a window if draw_ref_img is True (camera_image can be None)
    test_function: takes (sub_image, keys, trackbar_data0, trackbar_data1, ...) and returns an array of images
    win_name: name of window
    trackbar_info: iterator of (trackbar_name, trackbar_value, trackbar_max_value)
    write_to_video: if true, the raw camera input will be written to ./data/camera.avi
    draw_trackbars: if true, draw trackbar_info normally
        if false, use default value of trackbars as a constant and do not draw trackbars
    """
    cv.namedWindow(win_name)

    if draw_trackbars:
        for name, val, maxval in trackbar_info:
            cv.createTrackbar(name, win_name, val, maxval, lambda x: None)

    controls = get_control_positions("data/knobs_more.csv")
    if control_indices is not None:
        controls = [c for i, c in enumerate(controls) if i in control_indices]

    mouse_pos = (0, 0)

    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_pos
        mouse_pos = (x, y)

    cv.setMouseCallback(win_name, mouse_callback)


    out = None
    paused = False
    main_img, cam_img = None, None

    while True:
        if not paused:
            main_img, cam_img = image_producer()

        if write_to_video:
            if out is None:
                fourcc = cv.VideoWriter_fourcc(*'MJPG')
                out = cv.VideoWriter('./data/camera.avi',
                                     fourcc, 20.0, (cam_img.shape[1], cam_img.shape[0]))
            out.write(cam_img)

        if draw_ref_img:
            if main_img is not None:
                draw_resized(main_img, "transformed_reference_image", 0.6)
                # cv.imshow("reference_image", main_img)
            if cam_img is not None:
                cam_resized = cv.resize(
                    cam_img, (int(cam_img.shape[1]*0.5), int(cam_img.shape[0]*0.5)))
                # cv.imshow("camera_image", cam_resized)

        if draw_trackbars:
            trackbar_data = tuple(cv.getTrackbarPos(name, win_name)
                                  for name, _, _ in trackbar_info)
        else:
            trackbar_data = tuple(
                default_val for _, default_val, _ in trackbar_info)

        keys = cv.waitKey(20) & 0xff
        if keys == ord('q'):
            cv.destroyAllWindows()
            break
        elif keys == ord('p'):
            paused = not paused
        elif keys == ord('m'):
            print(f"{mouse_pos[0]}, {mouse_pos[1]}")


        # Get the test output for each controller test input
        # This will be an array of (array of image)
        test_outputs = [test_function(sub_image(
            main_img, control_box), keys, *trackbar_data) for control_box in controls]

        # get the max width of a column, for padding
        max_widths = [
            max(test_outputs, key=lambda x: x[col_index].shape[1])[
                col_index].shape[1]
            for col_index in range(len(test_outputs[0]))
        ]

        # Pad columns and concatenate rows
        test_outputs_padded = [
            np.concatenate([pad_to_width(img, max_widths[col_index])
                           for col_index, img in enumerate(test_output)], axis=1)
            for test_output in test_outputs
        ]

        combined = np.concatenate(test_outputs_padded, axis=0)

        #  print(lines.shape)
        cv.imshow(win_name, combined)

    if write_to_video:
        out.release()


def control_detect_test_static(input_image_name, *args, **kwargs):
    """ calls control_detect_tests, but with image_producer set to read from file, all other args identical"""
    main_img = cv.imread(input_image_name, -1)
    control_detect_test(lambda: main_img, *args, **kwargs)


def control_detect_test_recorded_video(video_file_name, *args, **kwargs):
    """ run control_detect_test, but read from video file instead of webcam """
    cap = cv.VideoCapture(video_file_name)
    transform_cache = np.zeros((490, 1650, 3), dtype=np.uint8)

    def image_producer():
        nonlocal cap, transform_cache
        ret, frame = cap.read()

        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        success, transformed, contour_img, _ = transform(
            frame, draw_debug=True)
        draw_resized(contour_img, "contours", 0.5)
        if success:
            transform_cache = transformed
        return transform_cache, frame

    control_detect_test(image_producer, *args, **kwargs)


def control_detect_test_video(*args, **kwargs):
    """ calls control_detect_tests, but with image_producer set to read from webcam, all other args identical"""
    cam_info = setup_camera(
        1280, 720, "data/camera_matrix_720.npy", "data/distortion_coeff_720.npy")

    transform_cache = np.zeros((490, 1650, 3), dtype=np.uint8)

    def image_producer():
        nonlocal cam_info, transform_cache
        frame = preprocess_frame(cam_info)
        success, transformed, contour_img, _ = transform(
            frame, draw_debug=True)
        draw_resized(contour_img, "contours", 0.7)
        if success:
            transform_cache = transformed
        return transform_cache, frame

    control_detect_test(image_producer, *args, **kwargs)

    # cam_info[0] is the VideoCapture object
    cam_info[0].release()
