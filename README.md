# rescuedrone-ml
Machine learning aspect of the Rescue Drone project.

`object_detection_test.py` is used to test the object detection model on either a live camera feed or an input video. This script can also be used to write the video output to file.

`create_tf_record.py` can be used to generate training record files for training.

## Requirements
- [Python 3.6](https://www.python.org/downloads/)
- [Google Cloud SDK](https://cloud.google.com/sdk/)
- [Tensorflow 1.5](https://www.tensorflow.org/install/)
- [Tensorflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

## Instructions
### Object Detection

Run the object detection test script using:

``` bash
python3 object_detection_test.py [-f file_path] [-s] [-w] [-t threshold]
```

Note that the `-s` flag is necessary in the command to display the video output and that the `-w` flag is necessary to write the video output to file.

If `frozen_inference_graph.pb` is not found within the `data/model/` directory, this script will automatically download and extract the default `ssd_mobilenet_v1_coco` training model.

### Creating Training Records

Start by placing images inside `input/images/` and annotations inside `input/annotations/`. Create `trainval.txt` by executing the following command within the main directory (`rescuedrone-ml/`):

``` bash
ls input/images/ | grep ".jpg" | sed s/.jpg// > input/annotations/trainval.txt
```

Training records can then be created by running:

``` bash
python3 create_tf_record.py --data_dir=input/ --output_dir=data/
```

Records will be generated within the `data/` directory.

### Training Object Detection Model using Google Cloud

Before training, make sure that you have executed the object detection test script at least once to generate the `model/` directory within `data/`.

Copy the `research/` directory from `tensorflow/models/research/` into the main directory.

Generate packages by running the command:

``` bash
python3 research/setup.py sdist
python3 research/slim/setup.py sdist
```

Set the `BUCKET_NAME` variable:

``` bash
export BUCKET_NAME="[BUCKET_NAME]"
```

Copy your data directory into your Google Cloud storage bucket with:

``` bash
gsutil -m cp -R data/ gs://$BUCKET_NAME/
```

For further training sessions, use instead:

``` bash
gsutil -m cp data/*.record gs://$BUCKET_NAME/data/
```

Start the training job by running the following command within the main directory:

``` bash
export JOB_TIME="$(date +%Y%m%d_%H%M%S)"
export JOB_NAME="train_$JOB_TIME"

gcloud ml-engine jobs submit training `whoami`_object_detection_`date +%s` \
    --runtime-version 1.2 \
    --job-dir=gs://$BUCKET_NAME/$JOB_NAME \
    --packages research/dist/object_detection-0.1.tar.gz,research/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region us-central1 \
    --config research/object_detection/samples/cloud/cloud.yml \
    -- \
    --train_dir=gs://rescuedrone-ml-bucket/$JOB_NAME \
    --pipeline_config_path=gs://rescuedrone-ml-bucket/data/ssd_mobilenet_v1_coco.config
```

Once training has been completed, set the `CHECKPOINT_NUMBER` variable to the checkpoint number of the model that you want to extract.

``` bash
export CHECKPOINT_NUMBER="[CHECKPOINT_NUMBER]"
```

Run the following within the main directory to extract the training model from the storage bucket:

``` bash
gsutil cp gs://$BUCKET_NAME/$JOB_NAME/model.ckpt-$CHECKPOINT_NUMBER.* train/
python research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path data/ssd_mobilenet_v1_coco.config \
    --trained_checkpoint_prefix $JOB_NAME/model.ckpt-$CHECKPOINT_NUMBER \
    --output_directory $JOB_NAME/

cp $JOB_NAME/frozen_inference_graph.pb data/model/
```
