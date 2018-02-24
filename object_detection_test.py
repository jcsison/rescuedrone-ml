import cv2
import datetime
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("./utils/")

import label_map_util
import visualization_utils as vis_util

show = False # set to True to display the output
write = False # set to True to write video output to file

video_input = 'input/DJI_0014.MOV' # Replace with path of input file or with 0 for camera
cap = cv2.VideoCapture(video_input)
cap_width = int(cap.get(3))
cap_height = int(cap.get(4))

if write:
    out = cv2.VideoWriter('output/output_{:%y%m%d_%H%M%S}.avi'.format(datetime.datetime.now()), cv2.VideoWriter_fourcc(*'XVID'), cap.get(5), (cap_width, cap_height))

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT =  'data/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 91

if not os.path.isfile(PATH_TO_CKPT):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            file.name = file_name
            tar_file.extract(file, 'data')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
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
        while True:
            ret, image_np = cap.read()
            if not ret:
                cap.release()
                print("Released video source.")
                break
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Testing class detection
            for i in range(min(20, np.squeeze(boxes).shape[0])):
                class_value = np.squeeze(classes).astype(np.int32)[i]
                score_value = np.squeeze(scores)[i]
                if class_value in category_index.keys():
                    ymin, xmin, ymax, xmax = tuple(np.squeeze(boxes)[i].tolist())
                    class_name = category_index[class_value]['name']
                    if score_value > 0.25:
                        center = (cap_width * (xmin + xmax) / 2, cap_height * (ymin + ymax) / 2)
                        print('{0} @ {1:.0f}%: xmin = {2:.1f}, xmax = {3:.1f}, ymin = {4:.1f}, ymax = {5:.1f}, center = ({6:.1f}, {7:.1f})'.format(class_name, score_value * 100, xmin * cap_width, xmax * cap_width, ymin * cap_height, ymax * cap_height, center[0], center[1]))

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=0.25,
                agnostic_mode=True,
                line_thickness=4)

            if write:
                out.write(image_np)

            if show:
                image_np = cv2.resize(image_np, (800, 600))
                cv2.imshow('Object Detection', image_np)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    except KeyboardInterrupt:
        cap.release()
        out.release()
        print("Released video source.")
