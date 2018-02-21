# rescuedrone-ml
Machine learning aspect of the Rescue Drone project.

`object_detection_test.py` is used to test the object detection model on either a live camera feed or an input video. This script can also be used to write the video output to file.

`create_tf_record.py` can be used to generate training record files for training.

## Requirements
- [Python 3.6](https://www.python.org/downloads/)
- [Tensorflow](https://www.tensorflow.org/install/)

## Instructions
### Object Detection

Follow [these steps](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) to install the Tensorflow Object Detection API.

Run the object detection test script using:

``` bash
python3 object_detection_test.py
```

Note that within the script the `show` variable must be set to `True` to display the video output and the `write` variable must be set to `True` to write the video output to file.

If `frozen_inference_graph.pb` is not found within the `data/` directory, this script will also automatically download and extract the default `ssd_mobilenet_v1_coco` training model.

### Creating Training Records

Create record files for training by placing images in `input/images/` and annotations in `input/annotations/` and executing:

``` bash
python3 create_tf_record.py --data_dir=data --output_dir=data
```

Training records will be generated within the `data/` directory.
