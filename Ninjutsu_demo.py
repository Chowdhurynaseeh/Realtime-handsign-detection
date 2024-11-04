#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import time
import copy
from collections import deque

import cv2 as cv
import numpy as np
from utils.args import get_args
from utils import CvFpsCalc, CvDrawText
from model.yolox.yolox_onnx import YoloxONNX

def initialize_camera(device, width, height):
    """Initialize the camera and set its properties."""
    cap = cv.VideoCapture(device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def load_model(model_path, input_shape, score_th, nms_th, nms_score_th, with_p6):
    """Load the YOLOX model."""
    return YoloxONNX(
        model_path=model_path,
        input_shape=input_shape,
        class_score_th=score_th,
        nms_th=nms_th,
        nms_score_th=nms_score_th,
        with_p6=with_p6,
        providers=['CUDAExecutionProvider']
    )

def load_labels_and_jutsu():
    """Load labels and jutsu data from CSV files."""
    with open('setting/labels.csv', encoding='utf8') as f:
        labels = [row for row in csv.reader(f)]
    
    with open('setting/jutsu.csv', encoding='utf8') as f:
        jutsu = [row for row in csv.reader(f)]
    
    return labels, jutsu

def process_frame(frame, yolox, score_th, sign_display_queue, sign_history_queue, chattering_check_queue, stable_list, stable_time):
    """Process a single frame and detect hand signs."""
    bboxes, scores, class_ids = yolox.inference(frame)
    for _, score, class_id in zip(bboxes, scores, class_ids):
        class_id = int(class_id) + 1
        if score < score_th:
            continue

        chattering_check_queue.append(class_id)
        if len(set(chattering_check_queue)) != 1:
            continue

        if len(sign_display_queue) == 0 or sign_display_queue[-1] != class_id:
            stable_list.append(class_id)

        if len(stable_list) > 0 and stable_list[-1] == class_id:
            stable_list.append(class_id)
            if len(stable_list) == stable_time:
                sign_display_queue.append(class_id)
                sign_history_queue.append(class_id)
                stable_list.clear()

def main():
    args = get_args()
    
    # Camera setup
    cap = initialize_camera(args.device, args.width, args.height)
    
    # Load model
    yolox = load_model(
        args.model, tuple(map(int, args.input_shape.split(','))),
        args.score_th, args.nms_th, args.nms_score_th, args.with_p6
    )

    # Load labels and jutsu
    labels, jutsu = load_labels_and_jutsu()
    
    # FPS calculation
    cvFpsCalc = CvFpsCalc()
    
    # Initialize variables
    font_path = './utils/font/衡山毛筆フォント.ttf'
    sign_display_queue = deque(maxlen=18)
    sign_history_queue = deque(maxlen=44)
    effect_images_queue = deque()
    chattering_check_queue = deque(maxlen=args.chattering_check)
    stable_list = []
    
    # Set default values for time tracking
    sign_interval_start = 0
    jutsu_index = 0
    jutsu_start_time = 0
    frame_count = 0
    window_name = 'HandSignDetection Demo'
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame)

        frame_count += 1
        if (frame_count % (args.skip_frame + 1)) != 0:
            continue

        # Process frame and detect signs
        process_frame(
            frame, yolox, args.score_th, sign_display_queue,
            sign_history_queue, chattering_check_queue, stable_list, 16
        )
        
        # Check jutsu and display results
        jutsu_index, jutsu_start_time = check_jutsu(
            sign_history_queue, labels, jutsu, jutsu_index, jutsu_start_time
        )
        
        # Draw debug information
        debug_image = draw_debug_image(
            debug_image, font_path, cvFpsCalc.get(), labels, bboxes, scores,
            class_ids, args.score_th, args.erase_bbox, args.use_display_score,
            jutsu, sign_display_queue, effect_images_queue, 18,
            args.jutsu_display_time, 16, 1, jutsu_index, jutsu_start_time
        )
        
        if args.use_fullscreen:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

        cv.imshow(window_name, debug_image)
        if cv.waitKey(1) == 27:  # ESC key to exit
            break

        time.sleep(max(0, (1.0 / args.fps) - (time.time() - sign_interval_start)))

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
