import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np

def read_file(file_list):
    '''
        read tfrecord files
    '''
    def _get_num(filepath):
        num=0
        t1 = time.time()
        for record in tf.python_io.tf_record_iterator(filepath):
            num=num+1
        t2 = time.time()
        print("{}s taken to get the number of files in tfrecord.".format(str(t2-t1)))
        return num

    # num_files=_get_num(filename)
    filename_queue = tf.train.string_input_producer(file_list)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features = {
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.string),
        'image/filepath': tf.FixedLenFeature([], tf.string),})

    images=tf.cast(features['image/encoded'], tf.string)
    images=tf.image.decode_image(images)
    labels = tf.cast(features['image/class/label'], tf.string)
    filenames = tf.cast(features['image/filepath'], tf.string)

    return images, labels, filenames

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

def predict(file_list):
    # get label
    label_map = get_label()

    # read tfrecord file
    images, labels, filenames = read_file(file_list)
    # image_batch, file_batch = tf.train.batch([image_data, filepath],batch_size=8)
    batch_size = 10
    capacity = 5000 + 3 * batch_size
    images = tf.image.resize_images(images, [331, 331])

    image_batch, filename_batch = tf.train.batch(
        [images, filenames],
        batch_size = batch_size,
        capacity = capacity,
    )

    # session
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        t1 = time.time()
        image_data_list, file_list, res = [], [], []
        # [input_0, _label, _filepath] = sess.run([image_data, label, filepath])
        '''
        for i in range(num_files):
            # image_data = tf.image.resize_images(image_data, [331, 331])
            t_a1 = time.time()
            [image, file] = sess.run([image_data, filepath])
            t_a2 = time.time()
            # print("sess run time: {}".format(t_a2-t_a1))
            image = tf.image.resize_images(image, [331, 331])
            image = sess.run(image)
            image_data_list.append(image)
            file_list.append(file)
            if (i != 0 and i % batch_size == 0) or (i == num_files-1):
                input = np.asarray(image_data_list)
                input = input.reshape(-1, 331, 331, 3)
                t_b1 = time.time()
                with tf.device('/gpu:0'):
                    predictions = inference_session.run(output_layer, feed_dict={input_layer: input})
                predictions = np.squeeze(predictions)
                t_b2 = time.time()

                print("inference_session #{} run time: {}".format(i, t_b2-t_b1))
                overall_result = np.argmax(np.sum(predictions, axis=0))
                # print("{}: {}".format(i, overall_result))
                # res.append(np.argmax(np.sum(prediction, axis=0)) for prediction in predictions)
                image_data_list = []
        '''
        try:
            while not coord.should_stop():
                with tf.device('/gpu:0'):
                    predictions = inference_session.run(output_layer, feed_dict={input_layer: image_batch})
                print(predictions)
        except:
            pass

        t2 = time.time()
        print("average speed: {}s/image".format((t2-t1)/num_files))


if __name__ == '__main__':
    model = "./model/nasnet_large_v1.pb"
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
    initializer = np.zeros([1, 331, 331, 3])
    inference_session.run(output_layer, feed_dict={input_layer: initializer})
    file_list = []
    for path, dir, files in os.walk("./input_data"):
        for file in files:
            if file == '.DS_Store': continue
            print("Reading file {}".format(file))
            file_path = os.path.join('./input_data', file)
            file_list.append(file_path)
            predict(file_list)
