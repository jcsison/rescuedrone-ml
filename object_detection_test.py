import os
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

import argparse
import cv2
import datetime
import numpy as np
import imutils
import tarfile
import random
import six.moves.urllib as urllib
import tensorflow as tf
import zipfile

from collections import defaultdict
from imutils.video import FileVideoStream
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from queue import Queue
from threading import Thread

sys.path.append('utils/')
sys.path.append('research/')
sys.path.append('research/object_detection/utils/')

import label_map_util
import visualization_utils as vis_util

class FileVideoStream:
    def __init__(self, path, queueSize=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        counter = 0
        while True:
            counter += 1
            if self.stopped:
                return
            if not self.Q.full():
                (grabbed, frame) = self.stream.read()
                if not grabbed:
                    self.stop()
                    return
                if counter % 20 == 0:
                    self.Q.put(frame)

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True

    def get_stream(self):
        return self.stream

def detect(video_input=0, show=False, write=False, threshold=0.5, process=False):
    def angle(coord):
        x, y = coord[0], coord[1]
        if x != 0:
            theta = np.rad2deg(abs(np.arctan(y / x)))
        else:
            theta = 90
        if x < 0 and y < 0:
            theta = theta + 90
        elif x < 0 and y >= 0:
            theta = theta + 180
        elif x >= 0 and y >= 0:
            theta = theta + 270
        return theta

    def direction(theta):
        if theta <= 9 or theta >= 351:
            direction = 'E'
        elif theta >= 81 and theta <= 99:
            direction = 'N'
        elif theta >= 171 and theta <= 189:
            direction = 'W'
        elif theta >= 261 and theta <= 279:
            direction = 'S'
        elif theta > 9 and theta < 81:
            direction = 'NE'
        elif theta > 99 and theta < 171:
            direction = 'NW'
        elif theta > 189 and theta < 261:
            direction = 'SW'
        else:
            direction = 'SE'
        return direction

    def next_seed():
        blur = round(1.5*random.random())*2 + 1
        tilt = round(360*random.random())
        erode = round(1*random.random()) + 1
        dilate = round(1*random.random()) + 1
        return blur, tilt, erode, dilate

    if (video_input == '0'):
        fvs = FileVideoStream(0).start()
    else:
        fvs = FileVideoStream(video_input).start()

    cap = fvs.get_stream()
    cap_width = int(cap.get(3))
    cap_height = int(cap.get(4))

    ref = (0, 0, 0, 0)

    if write:
        out = cv2.VideoWriter('output/output_{:%y%m%d_%H%M%S}.avi'.format(
            datetime.datetime.now()), cv2.VideoWriter_fourcc(*'XVID'), cap.get(5),
            (cap_width // 3, cap_height // 3))

    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    PATH_TO_CKPT =  os.path.join('data', 'model', 'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 1 # todo: change to 1

    if not os.path.isfile(PATH_TO_CKPT):
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if not file.isdir() and not 'saved_model.pb' in file_name:
                file.name = file_name
                tar_file.extract(file, os.path.join('data', 'model'))

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        try:
            init = False
            count = 0
            center_curr = np.array([0, 0])
            center_delta_curr = np.array([0, 0])
            center_ref = np.array([0.5, 0.5])
            theta = 0

            while True:
                image_np = fvs.read()
                image_np = imutils.resize(image_np, width=cap_width//3)

                if not fvs.more():
                    fvs.stop()
                    print("Released video source.")
                    break

                if process:
                    blur, tilt, erode, dilate = next_seed()
                    img_ycrcb = cv2.cvtColor(image_np, cv2.COLOR_BGR2YCR_CB)
                    channels = cv2.split(img_ycrcb)
                    channels[0] = cv2.equalizeHist(channels[0])
                    img_ycrcb = cv2.merge(channels)
                    image_np = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)
                    image_np = cv2.GaussianBlur(image_np, (blur, blur), 0)
                    # (h, w) = image_np.shape[:2]
                    # image_np = cv2.copyMakeBorder(image_np, 500, 500, 500, 500, cv2.BORDER_WRAP)
                    # (h2, w2) = image_np.shape[:2]
                    # image_np = cv2.warpAffine(image_np, cv2.getRotationMatrix2D((w2 / 2, h2 / 2), tilt, 1.0), (w2, h2))
                    # image_np = image_np[int(h2//2 - h//2):int(h2//2 + h/2), int(w2//2 - w/2):int(w2//2 + w/2)]
                    # image_np = cv2.erode(image_np, np.ones((erode, erode)))
                    # image_np = cv2.dilate(image_np, np.ones((dilate, dilate)))

                image_np_expanded = np.expand_dims(image_np, axis=0)
                (boxes, scores, classes, num) = sess.run([detection_boxes,
                    detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                for i in range(min(20, np.squeeze(boxes).shape[0])):
                    class_value = np.squeeze(classes).astype(np.int32)[i]
                    score_value = np.squeeze(scores)[i]
                    if class_value in category_index.keys():
                        ymin, xmin, ymax, xmax = tuple(np.squeeze(boxes)[i].tolist())
                        class_name = category_index[class_value]['name']
                        if score_value > threshold:
                            center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
                            center_prev = center_curr
                            center_curr = center
                            center_delta_prev = center_delta_curr
                            center_delta_curr = center_curr - center_prev
                            distance = np.linalg.norm(center_ref - center)
                            if distance > 0.25 and count < 5 and init:
                                count += 1
                                scores[0][i] = 0
                            else:
                                count = 0
                                if distance > 0.0625 or not init:
                                    init = True
                                    center_ref = center_curr
                                    ref = (xmin, xmax, ymin, ymax)
                                    theta = angle(center_delta_curr)
                                print(('{0} @ {1:.5}% | angle: {2:.6} | direction: {3:2} | center: ({4[0]:.2f}, {4[1]:.2f}) | distance from ref: {5:.6}').format(class_name, str(score_value * 100), str(theta), direction(theta), center, str(distance)))

                # vis_util.visualize_boxes_and_labels_on_image_array(
                #     image_np,
                #     np.squeeze(boxes),
                #     np.squeeze(classes).astype(np.int32),
                #     np.squeeze(scores),
                #     category_index,
                #     use_normalized_coordinates=True,
                #     min_score_thresh=threshold,
                #     agnostic_mode=True,
                #     line_thickness=4,
                #     angle=theta,
                #     direction=direction(theta))
                cv2.rectangle(image_np, (int(ref[0] * cap_width // 3), int(ref[2] * cap_height // 3)), (int(ref[1] * cap_width // 3), int(ref[3] * cap_height // 3)), (255, 0, 0), 2)

                if write:
                    out.write(image_np)

                if show:
                    cv2.imshow('Object Detection', image_np)

                if cv2.waitKey(50) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        except KeyboardInterrupt:
            fvs.stop()
            if write:
                out.release()
            print("Released video source.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='python3 object_detection_test.py [-f file_path] [-s] [-w] [-t threshold]')
    parser.add_argument('-f', '--file', type=str, default='0', help='Video file path, omit for camera')
    parser.add_argument('-s', '--show', action='store_const', const=True, default=False, help='Show video output')
    parser.add_argument('-w', '--write', action='store_const', const=True, default=False, help='Write video output to file')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Threshold for detection')
    parser.add_argument('-p', '--process', action='store_const', const=True, default=False, help='Process video input')
    args = parser.parse_args()

    detect(video_input=args.file, show=args.show, write=args.write, threshold=args.threshold, process=args.process)
