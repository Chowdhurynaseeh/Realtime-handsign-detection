import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument("--file", type=str, default=None)

    parser.add_argument("--fps", type=int, default=30)
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
        default=0.7,
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

    parser.add_argument("--sign_interval", type=float, default=2.0)
    parser.add_argument("--jutsu_display_time", type=int, default=5)

    parser.add_argument("--use_display_score", type=bool, default=False)
    parser.add_argument("--erase_bbox", type=bool, default=False)
    parser.add_argument("--use_jutsu_lang_en", type=bool, default=False)

    parser.add_argument("--chattering_check", type=int, default=1)

    parser.add_argument("--use_fullscreen", type=bool, default=False)

    args = parser.parse_args()

    return args