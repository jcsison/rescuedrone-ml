# rescuedrone-ml
Machine learning aspect of the Rescue Drone project.

`object_detection_test.py` is used to test the object detection model on either a live camera feed or an input video. This script can also be used to write the video output to file.

`create_tf_record.py` can be used to generate training record files for training.

## Requirements
- [Python 3.6](https://www.python.org/downloads/)
- [Tensorflow 1.5](https://www.tensorflow.org/install/)
- [Tensorflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

## Instructions
### Object Detection

Run the object detection test script using:

``` bash
python3 object_detection_test.py
```

Note that within the script the `show` variable must be set to `True` to display the video output and the `write` variable must be set to `True` to write the video output to file.

If `frozen_inference_graph.pb` is not found within the `data/` directory, this script will automatically download and extract the default `ssd_mobilenet_v1_coco` training model.

### Creating Training Records

Start by placing images in `input/images/` and annotations in `input/annotations/`. Create `trainval.txt` by executing the following command:

``` bash
# From input/ directory
ls images | grep ".png" | sed s/.png// > annotations/trainval.txt
```

Training records can then be created by running:

``` bash
python3 create_tf_record.py --data_dir=data --output_dir=data
```

Records will be generated within the `data/` directory.
