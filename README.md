# nasnet-tensorflow

A nasnet in tensorflow based on tensorflow [slim](https://github.com/tensorflow/models/tree/master/research/slim) library.


## About Nasnet and this repository

Nasnet is so far the state-of-the-art image classification architecture ImageNet dataset (ArXiv release date is 21 Jul. 2017), the single crop accuracy for nasnet-large model is reported to be 82.7. For details of nasnet, please refer to paper [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) by Barret Zoph etc.

With this repo., you should be able to:

- Train a nasnet for customized dataset for image classification problem from scratch. (If you want)

- Finetune nasnet (nasnet-a-large, nasnet-a-mobile) from a pre-train model for customized dataset for image classification problem.

- Test and evaluate the model you have trained.

- Deploy the model for your application. (By your self)

Suitable for those who have solid CNN knowledge, python and tensorflow background. For those who have less background, [tensorflow slim walk through tutorial](https://github.com/tensorflow/models/blob/master/research/slim/slim_walkthrough.ipynb) should be a good start.


## Dependencies
tensorflow >= 1.4.0

tf.contrib.slim

numpy


## Code coming soon
