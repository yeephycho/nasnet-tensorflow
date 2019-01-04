from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
slim = tf.contrib.slim

MASTER = ''
TRAIN_DIR = './train'
NUM_CLONES = 1
CLONE_ON_CPU = False
WORKER_REPLICAS = 1
NUM_PS_TASKS = 0
NUM_READERS = 4
NUM_PREPROCESSING_THREADS = 4
LOG_EVERY_N_STEPS = 10
SAVE_SUMMARIES_SECS = 600
SAVE_INTERVAL_SECS = 600
TASK = 0
WEIGHT_DECAY = 0.00004
OPTIMIZER = 'rmsprop'
ADADELTA_RHO = 0.95
ADAGRAD_INITIAL_ACCUMULATOR_VALUE = 0.1
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
OPT_EPSILON = 1.0
FTRL_LEARNING_RATE_POWER = -0.5
FTRL_INITIAL_ACCUMULATOR_VALUE = 0.1
FTRL_L1 = 0.0
FTRL_L2 = 0.0
MOMENTUM = 0.9
RMSPROP_MOMENTUM = 0.9
RMSPROP_DECAY = 0.9
LEARNING_RATE_DECAY_TYPE = 'exponential'
LEARNING_RATE = 0.01
END_LEARNING_RATE = 0.0001
LABEL_SMOOTHING = 0.0
LEARNING_RATE_DECAY_FACTOR = 0.94
NUM_EPOCHS_PER_DECAY = 2.0
SYNC_REPLICAS = False
REPLICAS_TO_AGGREGATE = 1
MOVING_AVERAGE_DECAY = None
DATASET_NAME = 'customized'
DATASET_SPLIT_NAME = 'train'
DATASET_DIR = './input_data'
LABELS_OFFSET = 0
MODEL_NAME = 'nasnet_large'
PREPROCESSING_NAME = None
BATCH_SIZE = 16
TRAIN_IMAGE_SIZE = None
MAX_NUMBER_OF_STEPS = 100000
CHECKPOINT_PATH = None
CHECKPOINT_EXCLUDE_SCOPES = None
TRAINABLE_SCOPES = None
IGNORE_MISSING_VARS = False


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    # config model deploy
    with tf.Graph().as_default():
      deploy_config = model_deploy.DeploymentConfig(
          num_clones=NUM_CLONES,
          clone_on_cpu=CLONE_ON_CPU,
          replica_id=TASK,
          num_replicas=WORKER_REPLICAS,
          num_ps_tasks=NUM_PS_TASKS)

      # Create global_step
      with tf.device(deploy_config.variables_device()):
        global_step = slim.create_global_step()

      # get the dataset
      dataset = dataset_factory.get_dataset(
          DATASET_NAME, DATASET_SPLIT_NAME, DATASET_DIR)

      # get the network graph
      network_fn = nets_factory.get_network_fn(
          MODEL_NAME,
          num_classes=(dataset.num_classes - LABELS_OFFSET),
          weight_decay=WEIGHT_DECAY,
          is_training=True)

      # Select the preprocessing function
      preprocessing_name = PREPROCESSING_NAME or MODEL_NAME
      image_preprocessing_fn = preprocessing_factory.get_preprocessing(
          preprocessing_name,
          is_training=True)

      # Create a dataset provider that loads data from the dataset
      with tf.device(deploy_config.inputs_device()):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=NUM_READERS,
            common_queue_capacity=20 * BATCH_SIZE,
            common_queue_min=10 * BATCH_SIZE)
        [image, label] = provider.get(['image', 'label'])
        label -= LABELS_OFFSET

        train_image_size = TRAIN_IMAGE_SIZE or network_fn.default_image_size

        image = image_preprocessing_fn(image, train_image_size, train_image_size)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=BATCH_SIZE,
            num_threads=NUM_PREPROCESSING_THREADS,
            capacity=5 * BATCH_SIZE)
        labels = slim.one_hot_encoding(
            labels, dataset.num_classes - LABELS_OFFSET)
        batch_queue = slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=2 * deploy_config.num_clones)

      # Define the model
      def clone_fn(batch_queue):
        """Allows data parallelism by creating multiple clones of network_fn."""
        images, labels = batch_queue.dequeue()
        logits, end_points = network_fn(images)

        # Specify the loss function
        if 'AuxLogits' in end_points:
          slim.losses.softmax_cross_entropy(
              end_points['AuxLogits'], labels,
              label_smoothing=LABEL_SMOOTHING, weights=0.4,
              scope='aux_loss')
        slim.losses.softmax_cross_entropy(
            logits, labels, label_smoothing=LABEL_SMOOTHING, weights=1.0)
        return end_points

      # Gather initial summaries.
      summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

      clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
      first_clone_scope = deploy_config.clone_scope(0)
      # Gather update_ops from the first clone. These contain, for example,
      # the updates for the batch_norm variables created by network_fn.
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

      # Add summaries for end_points.
      end_points = clones[0].outputs
      for end_point in end_points:
        x = end_points[end_point]
        summaries.add(tf.summary.histogram('activations/' + end_point, x))
        summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                        tf.nn.zero_fraction(x)))

      # Add summaries for losses.
      for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
        summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

      # Add summaries for variables.
      for variable in slim.get_model_variables():
        summaries.add(tf.summary.histogram(variable.op.name, variable))

      # Configure the moving averages
      if FLAGS.moving_average_decay:
        moving_average_variables = slim.get_model_variables()
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, global_step)
      else:
        moving_average_variables, variable_averages = None, None

      #########################################
      # Configure the optimization procedure. #
      #########################################
      with tf.device(deploy_config.optimizer_device()):
        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate,
            global_step,
            decay_steps,
            FLAGS.learning_rate_decay_factor,
            staircase=True,
            name='exponential_decay_learning_rate')
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon)
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

      if FLAGS.sync_replicas:
        # If sync_replicas is enabled, the averaging will be done in the chief
        # queue runner.
        optimizer = tf.train.SyncReplicasOptimizer(
            opt=optimizer,
            replicas_to_aggregate=FLAGS.replicas_to_aggregate,
            total_num_replicas=FLAGS.worker_replicas,
            variable_averages=variable_averages,
            variables_to_average=moving_average_variables)
      elif FLAGS.moving_average_decay:
        # Update ops executed locally by trainer.
        update_ops.append(variable_averages.apply(moving_average_variables))

      # Variables to train.
      variables_to_train = tf.trainable_variables()

      #  and returns a train_tensor and summary_op
      total_loss, clones_gradients = model_deploy.optimize_clones(
          clones,
          optimizer,
          var_list=variables_to_train)
      # Add total_loss to summary.
      summaries.add(tf.summary.scalar('total_loss', total_loss))

      # Create gradient updates.
      grad_updates = optimizer.apply_gradients(clones_gradients,
                                               global_step=global_step)
      update_ops.append(grad_updates)

      update_op = tf.group(*update_ops)
      with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')

      # Add the summaries from the first clone. These contain the summaries
      # created by model_fn and either optimize_clones() or _gather_clone_loss().
      summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                         first_clone_scope))

      # Merge all summaries together.
      summary_op = tf.summary.merge(list(summaries), name='summary_op')

      # Kicks off the training
      slim.learning.train(
          train_tensor,
          logdir=FLAGS.train_dir,
          master=FLAGS.master,
          is_chief=(FLAGS.task == 0),
          init_fn=None,
          summary_op=summary_op,
          number_of_steps=FLAGS.max_number_of_steps,
          log_every_n_steps=FLAGS.log_every_n_steps,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs,
          sync_optimizer=optimizer if FLAGS.sync_replicas else None)


if __name__ == '__main__':
    tf.app.run(main=main)
