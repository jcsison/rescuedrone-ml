# rescuedrone-ml
Machine learning aspect of the Rescue Drone project.
`object_detection.py` is used to test the object detection model on either a live camera feed or an input video. `object_detection.py` can also write the video output to file.
`create_tf_record.py` is used to generate training record files for training.

## Requirements
- [Python 3.6](https://www.python.org/downloads/)
- [Tensorflow](https://www.tensorflow.org/install/)

## Instructions
### Object Detection

Follow [these steps](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) to install the Tensorflow Object Detection API

Run the object detection script using:

``` bash
python3 object_detection_test.py
```

Note that the `show` variable must be set to `True` to display the video output and that the `write` variable must be set to true to write the video output to file.

### Creating Training Records

Generate record files for training by placing images in `input/images/` and annotations in `input/annotations/` and executing:

``` bash
python3 create_tf_record.py --data_dir=data --output_dir=data
```

Training records will be created within the `data/` directory.
