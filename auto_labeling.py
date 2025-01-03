#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import deque
import csv
from distutils.log import debug
import time
import copy
import argparse
import os

import cv2 as cv
import xml_edit

from model.yolox.yolox_onnx import YoloxONNX


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--skip_frame", type=int, default=0)

    parser.add_argument(
        "--model",
        type=str,
        default='model/yolox/yolox_nano.onnx',
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default="416,416",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.6,
        help='Class confidence',
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.45,
        help='NMS IoU threshold',
    )
    parser.add_argument(
        '--nms_score_th',
        type=float,
        default=0.1,
        help='NMS Score threshold',
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()
    # cap_device = args.device
    # cap_width = args.width
    # cap_height = args.height
    fps = args.fps
    # skip_frame = args.skip_frame

    model_path = args.model
    input_shape = tuple(map(int, args.input_shape.split(',')))
    score_th = args.score_th
    nms_th = args.nms_th
    nms_score_th = args.nms_score_th
    with_p6 = args.with_p6

    if args.file is not None:
        cap_device = args.file

    frame_count = 0

    # Prepare camera ###############################################################
    # cap = cv.VideoCapture(cap_device)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Load model #############################################################
    yolox = YoloxONNX(
        model_path=model_path,
        input_shape=input_shape,
        class_score_th=score_th,
        nms_th=nms_th,
        nms_score_th=nms_score_th,
        with_p6=with_p6,
        # providers=['CPUExecutionProvider'],
    )

    # Load labels ###########################################################
    with open('setting/labels.csv', encoding='utf8') as f:
        labels = csv.reader(f)
        labels = [row for row in labels]

    # TODO: Path
    folders = ['rasengan', 'rat', 'snake', 'tiger']

    for folder in folders:
        path = '../dataset/'+folder
        image_path = deque(os.listdir(path))

        while len(image_path) > 0:
            start_time = time.time()

            filename = image_path.popleft()
            frame = cv.imread(os.path.join(path, filename))
            debug_image = copy.deepcopy(frame)
            height, width = (480, 640)

            # Perform detection #############################################################
            bboxes, scores, class_ids = yolox.inference(frame)

            # Miss detection
            miss = (len(class_ids) != 1)

            for bbox, score, class_id in zip(bboxes, scores, class_ids):
                class_id = int(class_id) + 1
                
                # Visualize detection results ###################################################
                x1, y1 = int(bbox[0]), int(bbox[1])
                x2, y2 = int(bbox[2]), int(bbox[3])
                
                xml_edit.xml_generate(path, folder, filename, width, height,
                                        x1, y1, x2, y2, miss)

                cv.putText(
                    debug_image, 'ID:' + str(class_id) + ' ' +
                    labels[class_id][0] + ' ' + '{:.3f}'.format(score),
                    (x1, y1 - 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    cv.LINE_AA)
                cv.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if miss:
                cv.imwrite(f'miss/{filename}', debug_image)

            # Key processing (ESC: exit) #################################################
            key = cv.waitKey(1)
            if key == 27:  # ESC
                break

            # FPS adjustment #############################################################
            elapsed_time = time.time() - start_time
            sleep_time = max(0, ((1.0 / fps) - elapsed_time))
            time.sleep(sleep_time)

            cv.putText(
                debug_image,
                "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
                (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

            # Display the image #############################################################
            cv.imshow('NARUTO HandSignDetection Simple Demo', debug_image)

        # cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
