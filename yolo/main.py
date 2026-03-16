import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# -------------------- ARGUMENT PARSER --------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model',
    help='Path to YOLO model file',
    required=True
)

parser.add_argument(
    '--source',
    help='Image / video / usb camera (usb0)',
    required=True
)

parser.add_argument(
    '--thresh',
    help='Confidence threshold',
    default=0.75
)

parser.add_argument(
    '--resolution',
    help='Resolution WxH (example: 1280x720)',
    default=None
)

parser.add_argument(
    '--record',
    help='Record output video',
    action='store_true'
)

args = parser.parse_args()

# -------------------- USER INPUTS --------------------

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# -------------------- MODEL CHECK --------------------

if not os.path.exists(model_path):
    print("ERROR: Model path invalid")
    sys.exit()

# Load YOLO model
model = YOLO(model_path)
labels = model.names

# -------------------- SOURCE TYPE --------------------

img_ext_list = ['.jpg','.jpeg','.png','.bmp']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'

elif os.path.isfile(img_source):

    _, ext = os.path.splitext(img_source)

    if ext.lower() in img_ext_list:
        source_type = 'image'

    elif ext.lower() in vid_ext_list:
        source_type = 'video'

    else:
        print("Unsupported file type")
        sys.exit()

elif 'usb' in img_source:

    source_type = 'usb'
    usb_idx = int(img_source[3:])

else:
    print("Invalid source")
    sys.exit()

# -------------------- RESOLUTION --------------------

resize = False

if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# -------------------- VIDEO RECORDING --------------------

if record:

    if not user_res:
        print("Recording requires resolution")
        sys.exit()

    recorder = cv2.VideoWriter(
        "demo.avi",
        cv2.VideoWriter_fourcc(*'MJPG'), # type: ignore
        30,
        (resW, resH)
    )

# -------------------- LOAD SOURCE --------------------

if source_type == 'image':

    imgs_list = [img_source]

elif source_type == 'folder':

    imgs_list = []

    files = glob.glob(img_source + '/*')

    for f in files:
        if os.path.splitext(f)[1].lower() in img_ext_list:
            imgs_list.append(f)

elif source_type in ['video','usb']:

    cap = cv2.VideoCapture(usb_idx if source_type=='usb' else img_source)

    if user_res:
        cap.set(3,resW)
        cap.set(4,resH)

# -------------------- COLORS --------------------

bbox_colors = [
(164,120,87),(68,148,228),(93,97,209),
(178,182,133),(88,159,106),(96,202,231),
(159,124,168),(169,162,241),(98,118,150),
(172,176,184)
]

# -------------------- FPS VARIABLES --------------------

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# -------------------- INFERENCE LOOP --------------------

while True:

    t_start = time.perf_counter()

    # -------- GET FRAME --------

    if source_type in ['image','folder']:

        if img_count >= len(imgs_list):
            print("Finished images")
            break

        frame = cv2.imread(imgs_list[img_count])
        img_count += 1

    else:

        ret, frame = cap.read()

        if not ret:
            print("Video finished")
            break

    if resize:
        frame = cv2.resize(frame,(resW,resH))

    # -------- YOLO INFERENCE --------

    results = model(frame, conf=min_thresh, iou=0.45, verbose=False)

    detections = results[0].boxes

    object_count = 0

    # -------- PROCESS DETECTIONS --------

    for det in detections:

        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy

        classidx = int(det.cls.item())
        classname = labels[classidx]

        conf = det.conf.item()

        # Remove extremely large boxes (false detection)
        box_area = (xmax-xmin)*(ymax-ymin)
        frame_area = frame.shape[0]*frame.shape[1]

        if box_area > frame_area*0.6:
            continue

        color = bbox_colors[classidx % 10]

        cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),color,2)

        label = f"{classname}: {conf:.2f}"

        labelSize,_ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )

        y = max(ymin,labelSize[1]+10)

        cv2.rectangle(
            frame,
            (xmin,y-labelSize[1]-10),
            (xmin+labelSize[0],y),
            color,
            cv2.FILLED
        )

        cv2.putText(
            frame,
            label,
            (xmin,y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,0),
            1
        )

        object_count += 1

    # -------- FPS --------

    if source_type in ['video','usb']:

        cv2.putText(
            frame,
            f"FPS: {avg_frame_rate:.2f}",
            (10,20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,255),
            2
        )

    # -------- OBJECT COUNT --------

    cv2.putText(
        frame,
        f"Objects: {object_count}",
        (10,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    cv2.imshow("YOLO Detection",frame)

    if record:
        recorder.write(frame)

    key = cv2.waitKey(5)

    if key == ord('q'):
        break

    # -------- FPS CALCULATION --------

    t_stop = time.perf_counter()

    fps = 1/(t_stop-t_start)

    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)

    frame_rate_buffer.append(fps)

    avg_frame_rate = np.mean(frame_rate_buffer)

# -------------------- CLEANUP --------------------

print(f"Average FPS: {avg_frame_rate:.2f}")

if source_type in ['video','usb']:
    cap.release()

if record:
    recorder.release()

cv2.destroyAllWindows()