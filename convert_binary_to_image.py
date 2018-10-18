# -*- coding: utf-8 -*-
import tensorflow as tf
from PIL import Image

from PIL import Image
import numpy as np
import tensorflow as tf

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                        features={
                                          'image_data': tf.FixedLenFeature([], tf.string),
                                          'name': tf.FixedLenFeature([], tf.int64),
                                        })
    image = tf.decode_raw(features['image_data'], tf.uint8)
    name = tf.cast(features['name'], tf.int32)
    return image, name


def get_all_records(FILE):
 with tf.Session() as sess:
   filename_queue = tf.train.string_input_producer([ FILE ])
   image, name = read_and_decode(filename_queue)
   image = tf.reshape(image, [331, 331, 3])
   image.set_shape([331,331,3])
   init_op = tf.initialize_all_variables()
   sess.run(init_op)
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(coord=coord)
   for i in range(2053):
     example, l = sess.run([image, label])
     img = Image.fromarray(example, 'RGB')
     img.save( "convert/" + str(i) + '.jpeg')

     print (example,l)
   coord.request_stop()
   coord.join(threads)

get_all_records('./flowers_train_00000-of-00005.tfrecord')