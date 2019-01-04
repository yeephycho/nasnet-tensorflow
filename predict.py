import os, datetime, cv2
import tensorflow as tf
import numpy as np

batch_size = 8

def get_num_of_files_in_tfrecord(filepath):
    num=0
    for record in tf.python_io.tf_record_iterator(filepath):
        num=num+1
    return num

def diagnose_image(inference_session, input_image):
    with tf.device('/gpu:0'):
        predictions = inference_session.run(output_layer, feed_dict={input_layer: input_image})
    # print('Predictions:', predictions)
    predictions = np.squeeze(predictions)
    print('Predictions:', predictions)
    return predictions

def predict(filename):
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

        image_data_list, label_list, filepath_list = [], [], []

        res = []

        for i in range(num_files):
            start = datetime.datetime.now()
            [input_0, _label, _filepath] = sess.run([image_data, label, filepath])
            _label=str(_label,encoding='utf-8')
            _filepath=str(_filepath,encoding='utf-8')
            print(_filepath, _label)
            filepath_list.append(_filepath)

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

            end = datetime.datetime.now()
            #print(image_path)
            print(overall_result, label_map[overall_result].split(":")[-1])
            print("Time cost: ", end - start, "\n")
            res.append([_filepath,label_map[overall_result]])

    return res


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

    for path, dir, files in os.walk("./input_data"):
        for file in files:
            if file == '.DS_Store': continue
            print("Reading file {}".format(file))
            file_path = os.path.join('./input_data', file)
            predict(file_path)
