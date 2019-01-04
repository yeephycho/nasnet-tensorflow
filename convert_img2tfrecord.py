# encoding=utf-8

import tensorflow as tf
import numpy as np
import os
from PIL import Image

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_to_tfexample(image_data,label,filepath):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded':bytes_feature(image_data),
        'image/class/label':bytes_feature(bytes(label,encoding='utf-8')),
        'image/filepath': bytes_feature(bytes(filepath, encoding='utf-8')),
    }))

def get_num_of_files_in_tfrecord(filepath):
    num=0
    for record in tf.python_io.tf_record_iterator(filepath):
        num=num+1
    return num

class TFRecord:
    tf_writer=None

    def image_data_writer_open(self,filename):

        self.tf_writer=tf.python_io.TFRecordWriter(filename)

    def image_data_writer_close(self):
        self.tf_writer.close()

    def image_data_write_jpeg(self,image_data,filepath,label='test'):

        if(image_data.dtype!=np.uint8):
            image_data=image_data.astype(np.uint8)

        image_placeholder=tf.placeholder(dtype=tf.uint8)
        encoded_image=tf.image.encode_png(image_placeholder)  ###为了保证无损压缩,之后可能换成jpeg,对读取来说没有影响

        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True

        with tf.Session() as sess:
            [_image_encode]=sess.run([encoded_image],
                                     feed_dict={image_placeholder : image_data})
        example=image_to_tfexample(_image_encode,label,filepath)
        self.tf_writer.write(example.SerializeToString())

        return None

    def image_data_read_jpg(self,filename):
        num_files=get_num_of_files_in_tfrecord(filename)
        filename_queue = tf.train.string_input_producer([filename])

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,features = {
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.string),
            'image/filepath': tf.FixedLenFeature([], tf.string),})

        image_data=tf.cast(features['image/encoded'], tf.string)
        image_data=tf.image.decode_jpeg(image_data)

        label = tf.cast(features['image/class/label'], tf.string)

        filepath = tf.cast(features['image/filepath'], tf.string)


        with tf.Session() as sess:
            init_op = tf.initialize_all_variables()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            image_data_list=[]
            label_list=[]
            filepath_list=[]

            for i in range(num_files):
                [_image_data, _label,_filepath] = sess.run([image_data, label,filepath])
                _label=str(_label,encoding='utf-8')
                _filepath=str(_filepath,encoding='utf-8')

                image_data_list.append(_image_data)
                label_list.append(label_list)
                filepath_list.append(_filepath)
        coord.request_stop()
        coord.join(threads)
        return image_data_list,label_list,filepath_list

if __name__== '__main__':
    filename = "bankcard.tfrecord"
    tf_data = TFRecord()
    tf_data.image_data_writer_open(filename)
    for path, dir, files in os.walk("./bankcard"):
        for file in files:
            image = Image.open(file)
            image_arr = np.array(image)
            tf_data.image_data_write_jpeg(image_arr, file, 'bankcard')
    tf_data.image_data_writer_close()
