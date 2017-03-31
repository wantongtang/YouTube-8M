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
