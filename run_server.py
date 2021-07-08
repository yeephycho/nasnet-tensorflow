# encoding:utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import re
import os
import sys
import tarfile
import argparse
import datetime
import time, cv2
import numpy as np
# import webp
import tensorflow as tf
from six.moves import urllib
from datasets import dataset_factory
from nets import nets_factory
import read_binary
from preprocessing import preprocessing_factory
from PIL import Image
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory


def get_num_of_files_in_tfrecord(filepath):
    num=0
    for record in tf.python_io.tf_record_iterator(filepath):
        num=num+1
    return num

def decode_image(jpeg_file):
    with tf.device('/cpu:0'):
        decoder_graph = tf.Graph()
        with decoder_graph.as_default():
            decoded_image = tf.image.decode_jpeg(jpeg_file)
            normalized_image = tf.divide(decoded_image, 255)
            # reshaped_image = tf.reshape(normalized_image, [-1, 331, 331, 3])
        with tf.Session(graph = decoder_graph) as image_session:
        # image_session = tf.Session(graph = decoder_graph)
            input_0 = image_session.run(normalized_image)
    return input_0


def diagnose_image(inference_session, input_image):
    with tf.device('/gpu:0'):
        predictions = inference_session.run(output_layer, feed_dict={input_layer: input_image})
    predictions = np.squeeze(predictions)
    return predictions


def image_data_read_jpg(filename, convert_folder):
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
            _filepath=str(os.path.basename(_filepath),encoding='utf-8')

            image_data_list.append(_image_data)
            label_list.append(label_list)
            filepath_list.append(_filepath)
            img=Image.fromarray(_image_data, 'RGB')
            img.save(convert_folder + '/' + _filepath)

    coord.request_stop()
    coord.join(threads)
    return image_data_list,label_list,filepath_list


def predict(src_root):
    """ Inference the whole src root directory """
    # src_root = "./demo/img"
    dst_root = "./result"
    label_map_path = "./labelmap/label.txt"
    if not os.path.isdir(dst_root):
        os.mkdir(dst_root)

    images = os.listdir(src_root)
    output_file = os.path.join(dst_root, "output_result.txt")
    result_file = open(output_file, "a")

    label_map_file = open(label_map_path)
    label_map = {}
    for line_number, label in enumerate(label_map_file.readlines()):
        label_map[line_number] = label[:-1]
        line_number += 1
    label_map_file.close()

    res = []
    for image in images:
        image_path = os.path.join(src_root, image)
        start = datetime.datetime.now()
        with tf.gfile.FastGFile(image_path, 'rb') as image_raw:
            image_file = image_raw.read()
            if 'jpg' or 'jpeg' in image_path:
                input_0 = decode_image(image_file)
                # print(input_0.shape)
                # print('this is a jpeg')
            elif 'png' in image_path:
                input_0 = decode_png(image_file)
                # print('this is a png')
            elif 'webp' in image_path:
                # print('this is a webp')
                input_0 = webp.load_image(image_file, 'RGBA')
            else:
                print('wrong file format')
                continue

            while input_0.shape[0] < 331 or input_0.shape[1] < 331:
                input_0 = cv2.pyrUp(input_0)
            while input_0.shape[0] >= 662 and input_0.shape[1] >= 662:
                input_0 = cv2.pyrDown(input_0)

            image_height = input_0.shape[0]
            # print(image_height)
            image_width = input_0.shape[1]
            # print(image_width)
            image_height_center = int(image_height/2)
            image_width_center = int(image_width/2)

            tl_crop = input_0[0:331, 0:331]
            tr_crop = input_0[0:331, image_width-331:image_width]
            bl_crop = input_0[image_height-331:image_height, 0:331]
            br_crop = input_0[image_height-331:image_height, image_width-331:image_width]
            center_crop = input_0[image_height_center - 165: image_height_center + 166, image_width_center - 165: image_width_center + 166]

            input_concat = np.asarray([tl_crop, tr_crop, bl_crop, br_crop, center_crop])
            # print(input_concat.shape)
            input_batch = input_concat.reshape(-1, 331, 331, 3)

        predictions = diagnose_image(inference_session, input_batch)
        # print(predictions)
        overall_result = np.argmax(np.sum(predictions, axis=0))

        # write result file
        result_file.write(image_path + "\n")
        result_file.write(str(overall_result) + "\n")

        # save img to the classified folder
        image_origin = cv2.imread(os.path.join(src_root, image))
        save_path = "save/"+str(overall_result)
        os.mkdir(save_path)
        save_file_path = os.path.join(save_path, image)
        if os.path.exists(save_file_path):
            save_file_path = os.path.join(save_path, image.split('.')[0]+"_dup"+image.split('.')[:-1])
        cv2.imwrite(os.path.join(save_path, image),image_origin)
        print("Image saved.")

        end = datetime.datetime.now()
        #print(image_path)
        print(overall_result, label_map[overall_result])
        print("Time cost: ", end - start, "\n")
        res.append([image_path,label_map[overall_result]])

    result_file.close()

    return res


def predict_tfrecord(filename):
    '''
        read label file
    '''
    label_map_path = "./labelmap/label.txt"
    label_map_file = open(label_map_path)
    label_map = {}
    for line_number, label in enumerate(label_map_file.readlines()):
        label_map[line_number] = label[:-1]
        line_number += 1
    label_map_file.close()
    
    # read tfrecord file
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
        # label_list=[]
        # filepath_list=[]
        res = []
        for i in range(num_files):
            start = datetime.datetime.now()
            [input_0, _label,_filepath] = sess.run([image_data, label,filepath])
            # _label=str(_label,encoding='utf-8')
            _filepath=str(_filepath,encoding='utf-8')
            
            while input_0.shape[0] < 331 or input_0.shape[1] < 331:
                input_0 = cv2.pyrUp(input_0)
            while input_0.shape[0] >= 662 and input_0.shape[1] >= 662:
                input_0 = cv2.pyrDown(input_0)

            image_height = input_0.shape[0]
            print(image_height)
            image_width = input_0.shape[1]
            print(image_width)
            image_height_center = int(image_height/2)
            image_width_center = int(image_width/2)

            tl_crop = input_0[0:331, 0:331]
            tr_crop = input_0[0:331, image_width-331:image_width]
            bl_crop = input_0[image_height-331:image_height, 0:331]
            br_crop = input_0[image_height-331:image_height, image_width-331:image_width]
            center_crop = input_0[image_height_center - 165: image_height_center + 166, image_width_center - 165: image_width_center + 166]

            input_concat = np.asarray([tl_crop, tr_crop, bl_crop, br_crop, center_crop])
            # print(input_concat.shape)
            input_batch = input_concat.reshape(-1, 331, 331, 3)

            predictions = diagnose_image(inference_session, input_batch)
            # print(predictions)
            overall_result = np.argmax(np.sum(predictions, axis=0))

            # save img to the classified folder
            '''
            image_origin = cv2.imread(os.path.join(src_root, image))
            save_path = "save/"+str(overall_result)
            os.mkdir(save_path)
            save_file_path = os.path.join(save_path, image)
            if os.path.exists(save_file_path):
                save_file_path = os.path.join(save_path, image.split('.')[0]+"_dup"+image.split('.')[:-1])
            cv2.imwrite(os.path.join(save_path, image),image_origin)
            print("Image saved.")
            '''
            end = datetime.datetime.now()
            #print(image_path)
            print(overall_result, label_map[overall_result])
            print("Time cost: ", end - start, "\n")
            res.append([_filepath,label_map[overall_result]])

    return res


from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './upload/'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'webp', 'tfrecord'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 上传单张照片并预测
@app.route('/single', methods=['GET','POST'])
def upload_single():
    save_folder_name = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], save_folder_name)
            os.mkdir(upload_folder)
            file.save(os.path.join(upload_folder, filename))
            res = predict(upload_folder)
            return jsonify(res)
    return "Upload Single File."


# 上传多张照片并预测
@app.route('/folder',methods = ['GET','POST'])
def upload_folder():
    save_folder_name = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], save_folder_name)
    os.mkdir(upload_folder)
    if request.method =='POST':
        files = request.files.getlist('file[]',None)
        print(files)
        if files:
            for file in files:
                filename = secure_filename(file.filename)
                file.save(os.path.join(upload_folder,filename))
            res = predict(upload_folder)
            return jsonify(res)
    return "Upload Folder."


# 上传一类照片并从中筛选出识别错误的照片
@app.route('/test', methods=['GET','POST'])
def test():
    save_folder_name = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], save_folder_name)
    os.mkdir(upload_folder)
    if request.method =='POST':
        files = request.files.getlist('file[]',None)
        print(files)
        if files:
            for file in files:
                filename = secure_filename(file.filename)
                file.save(os.path.join(upload_folder,filename))
            predict(upload_folder)
            return "Uploaded " + str(files)
    return "Upload Folder."


# 上传单个tfrecord文件
@app.route('/binary_single', methods=['GET', 'POST'])
def upload_binary_single():
    save_folder_name = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], save_folder_name)
    os.mkdir(upload_folder)
    if request.method =='POST':
        files = request.files.getlist('file',None)
        print(files)
        if files:
            for file in files:
                filename = secure_filename(file.filename)
                filepath = os.path.join(upload_folder,filename)
                file.save(filepath)
            res = predict_binary_single(filepath)
            return jsonify(res)
    return "Upload Folder."


# 上传tfrecord文件
@app.route('/binary', methods=['GET', 'POST'])
def upload_binary():
    save_folder_name = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], save_folder_name)
    os.mkdir(upload_folder)
    if request.method =='POST':
        files = request.files.getlist('file[]',None)
        print(files)
        if files:
            for file in files:
                filename = secure_filename(file.filename)
                filepath = os.path.join(upload_folder,filename)
                file.save(filepath)
                predict_tfrecord(filepath)
                convert_folder = os.path.join(upload_folder, 'convert')
                os.mkdir(convert_folder)
                image_data_read_jpg(filepath, convert_folder)
            res = predict(convert_folder)
            return jsonify(res)
    return "Upload Folder."

# 上传tfrecord文件
@app.route('/tfrecord', methods=['GET', 'POST'])
def tfrecord():
    save_folder_name = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], save_folder_name)
    os.mkdir(upload_folder)
    if request.method =='POST':
        files = request.files.getlist('file[]',None)
        print(files)
        if files:
            for file in files:
                filename = secure_filename(file.filename)
                filepath = os.path.join(upload_folder,filename)
                file.save(filepath)
                res = predict_tfrecord(filepath)
                
            # res = predict(convert_folder)
            return jsonify(res)
    return "Upload Folder."


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    model = "./inference/frozen_nasnet_large_v2.pb"
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

    app.run(
        host='0.0.0.0',
        port=6006
    )
