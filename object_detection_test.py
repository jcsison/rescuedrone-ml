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
        global counter
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
                if counter % 1 == 0:
                    frame = imutils.resize(frame, width=cap_width//3)
                    self.Q.put(frame)

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True

    def get_stream(self):
        return self.stream

def detection(video_input=0, show=False, write=False, threshold=0.5):
    def angle(coord):
        def unit_vector(vector):
            return vector / np.linalg.norm(vector)

        v1_u = unit_vector((1, 0))
        v2_u = unit_vector(coord)
        theta = np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
        if (coord[1] > 0):
            theta = 360 - theta
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

    if (video_input == '0'):
        fvs = FileVideoStream(0).start()
    else:
        fvs = FileVideoStream(video_input).start()

    cap = fvs.get_stream()
    global cap_width
    cap_width = int(cap.get(3))
    cap_height = int(cap.get(4))

    if write:
        out = cv2.VideoWriter('output/output_{:%y%m%d_%H%M%S}.avi'.format(
            datetime.datetime.now()), cv2.VideoWriter_fourcc(*'XVID'), cap.get(5),
            (cap_width // 3, cap_height // 3))

    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    PATH_TO_CKPT =  os.path.join('data', 'model', 'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 2

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
                theta = 0
                center_curr = np.array([0, 0])
                center_delta_curr = np.array([0, 0])
                center_ref = np.array([0.5, 0.5])
                center_prev = np.array([0.5, 0.5])
                ref = (0, 0, 0, 0)

                while True:
                    image_np = fvs.read()

                    if not fvs.more():
                        fvs.stop()
                        print("Released video source.")
                        break

                    if (counter % 20 == 0):
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
                                    center_curr = center
                                    center_delta_curr = center_curr - center_prev
                                    distance = np.linalg.norm(center_ref - center_curr)
                                    if distance > 0.25 and count < 5 and init:
                                        count += 1
                                        scores[0][i] = 0
                                    else:
                                        count = 0
                                        if distance > 0.1 or not init:
                                            init = True
                                            center_prev = center_ref
                                            center_ref = center_curr
                                            ref = (xmin, xmax, ymin, ymax)
                                            theta = angle(center_ref - center_prev)
                                        print(('{0} @ {1:.5}% | angle: {2:.6} | direction: {3:2} | center: ({4[0]:.2f}, {4[1]:.2f}) | distance from ref: {5:.6}').format(class_name, str(score_value * 100), str(theta), direction(theta), center, str(distance)))

                    coord_prev = (int(center_prev[0] * cap_width // 3), int(center_prev[1] * cap_height // 3))
                    coord_ref = (int(center_ref[0] * cap_width // 3), int(center_ref[1] * cap_height // 3))
                    min_vertex = (int(ref[0] * cap_width // 3), int(ref[2] * cap_height // 3))
                    max_vertex = (int(ref[1] * cap_width // 3), int(ref[3] * cap_height // 3))
                    cv2.rectangle(image_np, min_vertex, max_vertex, (255, 0, 0), 2)
                    cv2.arrowedLine(image_np, coord_prev, coord_ref, (255, 0, 0), 3, tipLength=0.25)
                    display_str = 'angle: {0:.2f} ({1})'.format(theta, direction(theta))
                    cv2.putText(image_np, display_str, (int(ref[0] * cap_width // 3), int((ref[2] - 0.0275) * cap_height // 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

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
    args = parser.parse_args()

    detection(video_input=args.file, show=args.show, write=args.write, threshold=args.threshold)
