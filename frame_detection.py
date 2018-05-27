import os
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

import cv2
import numpy as np
import tensorflow as tf

sys.path.append('utils/')
sys.path.append('research/')
sys.path.append('research/object_detection/utils/')

import label_map_util

def detect(image, threshold=0.5):
    PATH_TO_CKPT =  os.path.join('data', 'model', 'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 2
    objects = []

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

            image_np_expanded = np.expand_dims(image, axis=0)
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
                        objects.append([class_name, (xmin, xmax, ymin, ymax)])
    return objects

if __name__ == '__main__':
    image = cv2.imread('image.jpg')
    objects = detect(image, threshold=0.5)
    print(objects)
