import os, time, json
import tensorflow as tf
import numpy as np

batch_size = 32

def read_file(filename):
    '''
        read tfrecord files
    '''
    def _get_num(filepath):
        num=0
        t1 = time.time()
        for record in tf.python_io.tf_record_iterator(filepath):
            num=num+1
        t2 = time.time()
        # print("{}s taken to get the number of files in tfrecord.".format(str(t2-t1)))
        return num

    num_files=_get_num(filename)
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features = {
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.string),
        'image/filepath': tf.FixedLenFeature([], tf.string),
        }
    )

    image_data=tf.cast(features['image/encoded'], tf.string)
    image_data=tf.image.decode_image(image_data)
    label = tf.cast(features['image/class/label'], tf.string)
    filepath = tf.cast(features['image/filepath'], tf.string)

    return num_files, image_data, label, filepath

def get_label():
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

    return label_map

def predict(filename):
    # get label
    label_map = get_label()

    # read tfrecord file
    num_files, image_data, label, filepath = read_file(filename)
    # image_batch, file_batch = tf.train.batch([image_data, filepath],batch_size=8)

    # session
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        t1 = time.time()
        # image_data_list, filepath_list, res = [], [], []
        res = []
        for i in range(num_files):
            [_image, _label, _filepath] = sess.run([image_data, label, filepath])
            _label=str(_label,encoding='utf-8')
            _filepath=str(_filepath,encoding='utf-8')

            _image = tf.image.resize_images(_image, [331, 331])
            _image = sess.run(_image)
            _image = np.asarray([_image])
            _image = _image.reshape(-1, 331, 331, 3)

            with tf.device('/gpu:0'):
                predictions = inference_session.run(output_layer, feed_dict={input_layer: _image})
            predictions = np.squeeze(predictions)

            predictions = predictions.tolist()
            overall_result = predictions.index(max(predictions))
            print(overall_result)
            predict_result = label_map[overall_result].split(":")[-1]

            content = {}
            content['prob'] = str(np.max(predictions))
            content['label'] = predict_result
            content['filepath'] = _filepath
            res.append(content)


        t2 = time.time()
        print("average speed: {}s/image".format((t2-t1)/num_files))

    return res


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    model_dir = "./model"
    model = "nasnet_large_v1.pb"
    model_path = os.path.join(model_dir, model)
    model_graph = tf.Graph()
    with model_graph.as_default():
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            input_layer = model_graph.get_tensor_by_name("input:0")
            output_layer = model_graph.get_tensor_by_name('final_layer/predictions:0')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    inference_session = tf.Session(graph = model_graph, config=config)
    initializer = np.zeros([1, 331, 331, 3])
    inference_session.run(output_layer, feed_dict={input_layer: initializer})
    file_list = []
    processed_files = []
    for path, dir, files in os.walk("./input_data"):
        for file in files:
            if file == '.DS_Store': continue
            print("Reading file {}".format(file))
            file_path = os.path.join('./input_data', file)
            file_list.append(file_path)
            res = predict(file_path)
            processed_files.append(file)

    with open('./model_output/processed_files/test_{}_processed_files.json'.format(model), 'w') as f:
        f.write(json.dumps(processed_files))

    with open('./model_output/classify_result/test_{}_classify_result.json'.format(model), 'w') as f:
        f.write(json.dumps(res, indent=4, separators=(',',':')))
