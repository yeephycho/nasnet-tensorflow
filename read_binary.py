# encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def get_num_of_files_in_tfrecord(filepath):    #获取tfrecord中的文件数量
	num=0
	for record in tf.python_io.tf_record_iterator(filepath):
		num=num+1
	return num

def read_single(filepath):
    tfrecords_filename = "/tmp/data/flowers/lowers_train_00001-of-00005.tfrecord"

    filename_queue = tf.train.string_input_producer([tfrecords_filename],
                                                   num_epochs=1)
    # filename_queue = tf.train.string_input_producer([filepath],num_epochs=1)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'name': tf.FixedLenFeature([], tf.string),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    image = tf.decode_raw(features['img_raw'],tf.int64)
    image = tf.reshape(image, [331,331,3])
    print(image.shape)

    name = features['name']
    print('name: ', name)

    return image, name

def read_tfrecord(filepath):
    tfrecords_filename = "/tmp/data/flowers/lowers_train_00001-of-00005.tfrecord"

    # filename_queue = tf.train.string_input_producer([tfrecords_filename],num_epochs=1)
    filename_queue = tf.train.string_input_producer([filepath],num_epochs=1)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'name': tf.FixedLenFeature([], tf.string),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    image = tf.decode_raw(features['img_raw'],tf.int64)
    image = tf.reshape(image, [331,331,3])

    name = features['name']

    return image, name

if __name__=='__main__':
    filepath = '/root'
    read_single(filepath)
