import os
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

import argparse
import cv2
import datetime
import numpy as np
import tarfile
import six.moves.urllib as urllib
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append('utils/')

import label_map_util
import visualization_utils as vis_util

def detect(video_input=0, show=False, write=False, threshold=0.5):
    def angle(coord):
        x, y = coord[0], coord[1]
        if x != 0:
            theta = np.rad2deg(abs(np.arctan(y / x)))
        else:
            theta = 90
        if x < 0 and y >= 0:
            theta = theta + 90
        elif x < 0 and y < 0:
            theta = theta + 180
        elif x >= 0 and y < 0:
            theta = theta + 270
        return theta

    if (video_input == '0'):
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_input)
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
    NUM_CLASSES = 91

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
            center_curr = np.array([0, 0])
            center_delta_curr = np.array([0, 0])
            center_ref = np.array([0, 0])
            theta = 0
            while True:
                ret, image_np = cap.read()
                if not ret:
                    cap.release()
                    print("Released video source.")
                    break
                image_np = cv2.resize(image_np, (0, 0), fx=1/3, fy=1/3)
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
                            if np.linalg.norm(center_ref - center) > 0.05:
                                center_ref = center_curr
                                theta = angle(center_delta_curr)
                            print(('{0} @ {1:.5}% | angle: {2:.6} | center: ' +
                                '({3[0]:.2f}, {3[1]:.2f})').format(class_name,
                                str(score_value * 100), str(theta), center))

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    min_score_thresh=threshold,
                    agnostic_mode=True,
                    line_thickness=4,
                    angle=theta)

                if write:
                    out.write(image_np)

                if show:
                    cv2.imshow('Object Detection', image_np)

                if cv2.waitKey(50) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        except KeyboardInterrupt:
            cap.release()
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

    detect(video_input=args.file, show=args.show, write=args.write, threshold=args.threshold)
