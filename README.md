# YouTube-8M

### Sign up an account
Go to [Google Cloud ML Platform](https://cloud.google.com/ml-engine/) and sign up a free account.

### Set up a Google Cloud Project
1. [Create a new project](https://console.cloud.google.com/project?_ga=1.197152431.1429848579.1482169659): Click **Create Project** and follow the instructions. **NOTE**: A project named `youtube-8m-project` is created and used as the example in the rest of the tutorial.
2. [Enable the APIs](https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,dataflow,compute_component,logging,storage_component,storage_api,bigquery&_ga=1.205652499.1429848579.1482169659): After the APIs are enabled, **do not** click "Go to Credentials".

### Set up environment
1. Install Google Cloud SDK: [Mac](https://cloud.google.com/sdk/docs/quickstart-mac-os-x#before-you-begin), [Windows](https://cloud.google.com/sdk/docs/quickstart-windows#before-you-begin), [Linux](https://cloud.google.com/sdk/docs/quickstart-linux#before-you-begin)
2. Init Google Cloud SDK
```bash
gcloud init
```
3. Set up default project (Optinal if `gcloud init` already sets one.)
```bash
gcloud config set project youtube-8m-project
```
4. Install the latest version of TensorFlow
```bash
pip install -U pip
pip install -U --user tensorflow
```
**NOTE**: run `gcloud help` or `gcloud help [sub-command]` for more details.

### Run the starter code
1. Clone the `youtube-8m` repository
```bash
git clone https://github.com/google/youtube-8m.git
```
2. Run the code locally to make sure that it can function normally on Google Cloud
```
gcloud ml-engine local train \
    --package-path=youtube-8m --module-name=youtube-8m.train \
    --  --train_data_pattern='gs://youtube8m-ml/1/video_level/train/train*.tfrecord' \
        --train_dir=/tmp/yt8m_train --model=LogisticModel --start_new_model
```
3. Download dataset from Google Cloud, then train and test the model locally using the downloaded dataset (part of the whole set).
```bash
gsutil cp gs://us.data.yt8m.org/1/video_level/train/traina[0-9].tfrecord .
```
4. Once the model works well, run the model using the entire dataset on Google Cloud
    1. Create a storage bucket to store training logs and checkpoints. (Only for the first time)
   ```bash
   BUCKET_NAME=gs://${USER}_yt8m_train_bucket
   gsutil mb -l us-east1 $BUCKET_NAME
   ```
    2. Submit the training job    
   ```bash
   JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S)
   gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME \
      --package-path=youtube-8m --module-name=youtube-8m.train \
      --staging-bucket=$BUCKET_NAME --region=us-east1 \
      --config=youtube-8m/cloudml-gpu.yaml \
      --  --train_data_pattern='gs://youtube8m-ml-us-east1/1/video_level/train/train*.tfrecord' \
          --model=LogisticModel \
          --train_dir=$BUCKET_NAME/yt8m_train_video_level_logistic_model
   ```      
    3. Check the process of the submitted job in the [job console](https://console.cloud.google.com/ml/jobs)
    4. Also, check the training loss through [tensorboard](http://localhost:8080)
   ```bash
   tensorboard --logdir=$BUCKET_NAME --port=8080
   ```
**NOTE**: A model will be ***finetuned*** if the same `train_dir` is given. Use `--start_new_model` for training from scrach.
5. Evalute the model
```bash
JOB_TO_EVAL=yt8m_train_video_level_logistic_model
JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S)
gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME \
    --package-path=youtube-8m --module-name=youtube-8m.eval \
    --staging-bucket=$BUCKET_NAME --region=us-east1 \
    --config=youtube-8m/cloudml-gpu.yaml \
    --  --eval_data_pattern='gs://youtube8m-ml-us-east1/1/video_level/validate/validate*.tfrecord' \
        --model=LogisticModel \
        --train_dir=$BUCKET_NAME/${JOB_TO_EVAL} --run_once=True
```
6. Inference on the test dataset
```bash
JOB_TO_EVAL=yt8m_train_video_level_logistic_model
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S)
gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME \
    --package-path=youtube-8m --module-name=youtube-8m.inference \
    --staging-bucket=$BUCKET_NAME --region=us-east1 \
    --config=youtube-8m/cloudml-gpu.yaml \
    --  --input_data_pattern='gs://youtube8m-ml/1/video_level/test/test*.tfrecord' \
        --train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
        --output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions.csv
```
7. View the generated files on Google Cloud using [storage console](https://console.cloud.google.com/storage/browser)
8. Copy `predictions.csv` locally:
```bash
gsutil cp $BUCKET_NAME/${JOB_TO_EVAL}/predictions.csv .
```
9. Use frame level features:
```bash
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S)
gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME \
    --package-path=youtube-8m --module-name=youtube-8m.train \
    --staging-bucket=$BUCKET_NAME --region=us-east1 \
    --config=youtube-8m/cloudml-gpu.yaml \
    --  --train_data_pattern='gs://youtube8m-ml-us-east1/1/frame_level/train/train*.tfrecord' \
        --frame_features=True --model=FrameLevelLogisticModel --feature_names="rgb" \
        --feature_sizes="1024" --batch_size=128 \
        --train_dir=$BUCKET_NAME/yt8m_train_frame_level_logistic_model
```
10. Use audio features
    * Video-level
      ```bash
      --feature_names="mean_rgb, mean_audio" --feature_sizes="1024, 128"
      ```
    * Frame-level
      ```bash
      --feature_names="rgb, audio" --feature_sizes="1024, 128"
      ```
11. Download ground truth:
```bash
gsutil cp gs://us.data.yt8m.org/1/ground_truth_labels/train_labels.csv /destination/folder/
gsutil cp gs://us.data.yt8m.org/1/ground_truth_labels/validate_labels.csv /destination/folder/
```

### Evaluation Metric: Global Average Precision
![](http://www.sciweavers.org/tex2img.php?eq=GAP%20%3D%20%5Csum_%7Bi%3D1%7D%5EN%20p%28i%29%5CDelta%20r%28i%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

See the [code](https://github.com/google/youtube-8m/blob/master/average_precision_calculator.py#L179) for details.

### Dataset descriptions
#### Video-level data
Available on Google Cloud at `gs://us.data.yt8m.org/1/video_level/train`, `gs://us.data.yt8m.org/1/video_level/validate`, and `gs://us.data.yt8m.org/1/video_level/test`.

Each video has:
* "video_id": unique id for the video, in train set it is a Youtube video id, and in test/validation they are anonymized
* "labels": list of labels of that video
* "mean_rgb": float array of length 1024
* "mean_audio": float array of length 128
Example code:

```python
import tensorflow as tf

video_lvl_record = "video_level/train-1.tfrecord"
vid_ids, labels, mean_rgb, mean_audio = [], [], [], []

for example in tf.python_io.tf_record_iterator(video_lvl_record):
    tf_example = tf.train.Example.FromString(example)

    vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
    labels.append(tf_example.features.feature['labels'].int64_list.value)
    mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)
    mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)
```
#### Frame-level data
Available on Google Cloud at `gs://us.data.yt8m.org/1/frame_level/train`,  `gs://us.data.yt8m.org/1/frame_level/validate`, and `gs://us.data.yt8m.org/1/frame_level/test`.

Each video has:
* "video_id": unique id for the video
* "labels": list of labels of that video.
* Each frame has "rgb": float array of length 1024,
* Each frame has "audio": float array of length 128

Example code:
```python
import tensorflow as tf

frame_lvl_record = "frame_level/train-1.tfrecord"
feat_rgb, feat_audio = [], []

for example in tf.python_io.tf_record_iterator(frame_lvl_record):        
    tf_seq_example = tf.train.SequenceExample.FromString(example)
    n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)
    sess = tf.InteractiveSession()
    rgb_frame, audio_frame = [], []
    for i in range(n_frames):
        rgb_frame.append(tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0],tf.uint8)
                       ,tf.float32).eval())
        audio_frame.append(tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0],tf.uint8)
                       ,tf.float32).eval())
    sess.close()
    feat_rgb.append(rgb_frame)
    feat_audio.append(audio_frame)
```
#### Ground Truth
* train.csv: `gs://us.data.yt8m.org/1/ground_truth_labels/train_labels.csv`
* validation.csv: `gs://us.data.yt8m.org/1/ground_truth_labels/validate_labels.csv`

Each line has:
* VideoId - the id of the video
* Labels - the correct labels of the video (space delimited)

#### Others
* [label_names.csv](https://www.kaggle.com/c/youtube8m/download/label_names.csv.zip) - a mapping between label_id and label_name

* [sample_submission.csv](https://www.kaggle.com/c/youtube8m/download/sample_submission.csv.zip) - a sample submission file in the correct format， each line has:
    * VideoId - the id of the video
    * LabelConfidencePair - space delimited predictions and their probabilities

### Overview of Files in `youtube-8m`
#### Training
*   `train.py`: The primary script for training models.
*   `losses.py`: Contains definitions for loss functions.
*   `models.py`: Contains the base class for defining a model.
*   `video_level_models.py`: Contains definitions for models that take aggregated features as input.
*   `frame_level_models.py`: Contains definitions for models that take frame-level features as input.
*   `model_util.py`: Contains functions that are of general utility for implementing models.
*   `export_model.py`: Provides a class to export a model during training for later use in batch prediction.
*   `readers.py`: Contains definitions for the Video dataset and Frame dataset readers.
#### Evaluation
*   `eval.py`: The primary script for evaluating models.
*   `eval_util.py`: Provides a class that calculates all evaluation metrics.
*   `average_precision_calculator.py`: Functions for calculating average precision.
*   `mean_average_precision_calculator.py`: Functions for calculating mean average precision.
#### Inference
*   `inference.py`: Generates an output file containing predictions of the model over a set of videos.

#### Misc
*   `utils.py`: Common functions.
*   `convert_prediction_from_json_to_csv.py`: Converts the JSON output of batch prediction into a CSV file for submission.

### Inspirations
#### Relationship between different labels
![](https://www.kaggle.io/svf/863073/011c8874544a0a1f392ea08f5de9d37e/__results___files/__results___12_0.png)
#### Given features are not good enough
![](https://www.kaggle.io/svf/863073/011c8874544a0a1f392ea08f5de9d37e/__results___files/__results___13_0.png)

### Timeline
* **May 26, 2017** - Entry deadline. You must accept the competition rules before this date in order to compete.
* **May 26, 2017** - Team Merger deadline. This is the last day participants may join or merge teams.
* **June 2, 2017** - Final competition submission deadline.
* **June 16, 2017** - Paper submission deadline.
* **June 30, 2017** - Paper acceptance and final winners announcement.
* **July 14, 2017** - YouTube-8M Workshop camera-ready deadline.
* **July 26, 2017** - YouTube-8M Workshop at CVPR 2017.

### Look-up Table
* Set default project:
```bash
gcloud config set project [project_id]
```
* Man page for `gcloud`:
```bash
gcloud help` or `gcloud help [sub-command]
```
* Run code locally
```bash
gcloud ml-engine local train
    --package-path=[package_path] --module-name=[main_function_path]
    -- [params_for_main_function]
```
* Run code on Google Cloud:
```bash
gcloud --verbosity=[mode] ml-engine jobs submit training [job_name]
    --package-path=[package_path] --module-name=[main_function_path
    --staging-bucket=[log_dir] --region=[region] --config=[config_path]
    -- [params_for_main_function]
```
* Download data from Google Cloud:
```bash
gsutil cp [src_path] [dst_path]
```
* `mkdir` on Google Cloud:
```bash
gsutil mb -l [region] [folder_name]
```
* [Check job process](https://console.cloud.google.com/ml/jobs)
* TensorBoard:
```bash
tensorboard --logdir=[log_path] --port=[port_number]
```
* [View generated files](https://console.cloud.google.com/storage/browser)

### Supported python packages
* Tensorflow 1.0.1
* The following PyPI packages:
* numpy 1.12.1
* pandas 0.17.1
* scipy 0.17.0
* scikit-learn 0.17.0
* sympy 0.7.6.1
* statsmodels 0.6.1
* oauth2client 2.2.0
* httplib2 0.9.2
* python-dateutil 2.5.0
* argparse 1.2.1
* six 1.10.0
* PyYAML 3.11
* wrapt 1.10.8
* crcmod 1.7
* google-api-python-client 1.5.1
* python-json-logger 0.1.5
* gcloud 0.18.1
* subprocess32 3.2.7
* wheel 0.30.0a0
* WebOb 1.6.2
* Paste 2.0.3
* tornado 4.3
* grpcio 1.0.1
* requests 2.9.1
* webapp2 3.0.0b1
* google-cloud-logging 0.22.0
* The following Debian packages:
* curl
* libcurl3-dev
* wget
* zip
* unzip
* git
* vim
* build-essential
* ca-certificates
* pkg-config
* rsync
* libatlas-base-dev
* liblapack-dev
* gfortran
* python2.7
* python-dev
* python-setuptools
* gdb
* openjdk-8-jdk
* openjdk-8-jre-headless
* g++
* zlib1g-dev
* libio-all-perl
* module-init-tools
* libyaml-0-2
* python-opencv
