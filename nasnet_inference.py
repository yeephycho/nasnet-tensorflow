from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import re
import os
import sys
import tarfile
import argparse
import datetime
import numpy as np
import tensorflow as tf
from six.moves import urllib

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
model = "./inference/frozen_nasnet_large.pb"
model_graph = tf.Graph()
with model_graph.as_default():
    with tf.gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        input_layer = model_graph.get_tensor_by_name("input:0")
        output_layer = model_graph.get_tensor_by_name('final_layer/predictions:0')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
inference_session = tf.Session(graph = model_graph, config=config)

def decode_image(jpeg_file):
    with tf.device('/cpu:0'):
        decoder_graph = tf.Graph()
        with decoder_graph.as_default():
            decoded_image = tf.image.decode_jpeg(jpeg_file)
            resized_image = tf.image.resize_image_with_crop_or_pad(decoded_image,
                                                     331, 331)
            normalized_image = tf.divide(resized_image, 255)
            reshaped_image = tf.reshape(normalized_image, [-1, 331, 331, 3])
        with tf.Session(graph = decoder_graph) as image_session:
        # image_session = tf.Session(graph = decoder_graph)
            input_0 = image_session.run(reshaped_image)
    return input_0


def diagnose_image(inference_session, input_image):
    with tf.device('/gpu:0'):
        predictions = inference_session.run(output_layer, feed_dict={input_layer: input_image})
    predictions = np.squeeze(predictions)
    return predictions


def main(arguments):
    image_path = "./img/sunflower1.jpg"
    start = datetime.datetime.now()
    with tf.gfile.FastGFile(image_path, 'rb') as jpeg_file_raw:
        jpeg_file = jpeg_file_raw.read()
        input_0 = decode_image(jpeg_file)
    predictions = diagnose_image(inference_session, input_0)
    end = datetime.datetime.now()
    print(image_path)
    print(str(np.argmax(predictions)))
    print("Time spent: ", end - start)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
