# nasnet-tensorflow

A nasnet in tensorflow based on tensorflow [slim](https://github.com/tensorflow/models/tree/master/research/slim) library.


## About Nasnet and this repository

Nasnet is so far the state-of-the-art image classification architecture on ImageNet dataset (ArXiv release date is 21 Jul. 2017), the single crop accuracy for nasnet-large model is reported to be 82.7. For details of nasnet, please refer to paper [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) by Barret Zoph etc.

With this repo., you should be able to:

- Train a nasnet for customized dataset for image classification problem from scratch. (If you want)

- Finetune nasnet (nasnet-a-large, nasnet-a-mobile) from a pre-train model for image classification.

- Test and evaluate the model you have trained.

- Deploy the model for your application ore transfer the feature extractor part to other problems such as object detection. (By yourself)

Suitable for those who have solid CNN knowledge, python and tensorflow background. For those who have less background, [tensorflow slim walk through tutorial](https://github.com/tensorflow/models/blob/master/research/slim/slim_walkthrough.ipynb) should be a good start.


## Dependencies
tensorflow >= 1.4.0

tf.contrib.slim

numpy


## Usage
1. Clone the repo.
```shell
git clone https://github.com/yeephycho/nasnet-tensorflow.git
```

2. Download and converting to TFRecord format (This part is the same as tf.slim tutorial)
For each dataset, we'll need to download the raw data and convert it to
TensorFlow's native
[TFRecord](https://www.tensorflow.org/versions/r0.10/api_docs/python/python_io.html#tfrecords-format-details)
format. Each TFRecord contains a
[TF-Example](https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/core/example/example.proto)
protocol buffer. Below we demonstrate how to do this for the Flowers dataset.

```shell
$ DATA_DIR=/tmp/data/flowers
$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"
```

When the script finishes you will find several TFRecord files created:

```shell
$ ls ${DATA_DIR}
flowers_train-00000-of-00005.tfrecord
...
flowers_train-00004-of-00005.tfrecord
flowers_validation-00000-of-00005.tfrecord
...
flowers_validation-00004-of-00005.tfrecord
labels.txt
```

These represent the training and validation data, sharded over 5 files each.
You will also find the `$DATA_DIR/labels.txt` file which contains the mapping
from integer labels to class names.

3. Train from scratch
```shell
DATASET_DIR=/tmp/data/flowers
TRAIN_DIR=./train
```
```python
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=nasnet_mobile
```


## Code coming soon
