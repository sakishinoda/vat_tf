import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import sys

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/tmp/mnist',
                           'where to store the dataset')
tf.app.flags.DEFINE_integer('num_labeled_examples', 100, "The number of labeled examples")
tf.app.flags.DEFINE_integer('num_valid_examples', 50, "The number of validation examples")
tf.app.flags.DEFINE_integer('dataset_seed', 1, "dataset seed")

NUM_CLASSES = 10
NUM_EXAMPLES_TRAIN = 60000
NUM_EXAMPLES_TEST = 10000


def load_mnist():
    mnist = input_data.read_data_sets("MNIST_data/")
    images_train = np.concatenate([mnist.train.images, mnist.validation.images], 0).astype(np.float32)
    labels_train = np.concatenate([mnist.train.labels, mnist.validation.labels], 0).astype(np.int32)
    images_test = mnist.test.images.astype(np.float32)
    labels_test = mnist.test.labels.astype(np.int32)
    images_train = images_train
    images_test = images_test
    return (images_train, labels_train), (images_test, labels_test)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_images_and_labels(images, labels, filepath):
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (images.shape[0], num_examples))
    print('Writing', filepath)
    writer = tf.python_io.TFRecordWriter(filepath)
    for index in range(num_examples):
        image = images[index].tolist()
        image_feature = tf.train.Feature(float_list=tf.train.FloatList(value=image))
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(28),
            'width': _int64_feature(28),
            'depth': _int64_feature(1),
            'label': _int64_feature(int(labels[index])),
            'image': image_feature}))
        writer.write(example.SerializeToString())
    writer.close()


def read(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image': tf.FixedLenFeature([784], tf.float32),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    image = features['image']
    image = tf.reshape(image, [28, 28, 1])
    label = tf.one_hot(tf.cast(features['label'], tf.int32), 10)
    return image, label


def generate_batch(
        example,
        min_queue_examples,
        batch_size, shuffle):
    """
    Arg:
        list of tensors.
    """
    num_preprocess_threads = 1

    if shuffle:
        ret = tf.train.shuffle_batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        ret = tf.train.batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            allow_smaller_final_batch=True,
            capacity=min_queue_examples + 3 * batch_size)

    return ret


def transform(image):
    image = tf.reshape(image, [28, 28, 1])
    return image


def generate_filename_queue(filenames, data_dir, num_epochs=None):
    print("filenames in queue:", filenames)
    for i in range(len(filenames)):
        filenames[i] = os.path.join(data_dir, filenames[i])
    return tf.train.string_input_producer(filenames, num_epochs=num_epochs)


def prepare_dataset():
    (train_images, train_labels), (test_images, test_labels) = load_mnist()
    dirpath = os.path.join(FLAGS.data_dir, 'seed' + str(FLAGS.dataset_seed))
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    rng = np.random.RandomState(FLAGS.dataset_seed)
    rand_ix = rng.permutation(NUM_EXAMPLES_TRAIN)
    _train_images, _train_labels = train_images[rand_ix], train_labels[rand_ix]

    examples_per_class = int(FLAGS.num_labeled_examples / 10)
    labeled_train_images = np.zeros((FLAGS.num_labeled_examples, 784), dtype=np.float32)
    labeled_train_labels = np.zeros((FLAGS.num_labeled_examples), dtype=np.int64)
    for i in xrange(10):
        ind = np.where(_train_labels == i)[0]
        labeled_train_images[i * examples_per_class:(i + 1) * examples_per_class] \
            = _train_images[ind[0:examples_per_class]]
        labeled_train_labels[i * examples_per_class:(i + 1) * examples_per_class] \
            = _train_labels[ind[0:examples_per_class]]
        _train_images = np.delete(_train_images,
                                  ind[0:examples_per_class], 0)
        _train_labels = np.delete(_train_labels,
                                  ind[0:examples_per_class])

    rand_ix_labeled = rng.permutation(FLAGS.num_labeled_examples)
    labeled_train_images, labeled_train_labels = \
        labeled_train_images[rand_ix_labeled], labeled_train_labels[rand_ix_labeled]

    convert_images_and_labels(labeled_train_images,
                              labeled_train_labels,
                              os.path.join(dirpath, 'labeled_train.tfrecords'))
    convert_images_and_labels(train_images, train_labels,
                              os.path.join(dirpath, 'unlabeled_train.tfrecords'))
    convert_images_and_labels(test_images,
                              test_labels,
                              os.path.join(dirpath, 'test.tfrecords'))

    # Construct dataset for validation
    train_images_valid, train_labels_valid = \
        labeled_train_images[FLAGS.num_valid_examples:], labeled_train_labels[FLAGS.num_valid_examples:]
    test_images_valid, test_labels_valid = \
        labeled_train_images[:FLAGS.num_valid_examples], labeled_train_labels[:FLAGS.num_valid_examples]
    unlabeled_train_images_valid = np.concatenate(
        (train_images_valid, _train_images), axis=0)
    unlabeled_train_labels_valid = np.concatenate(
        (train_labels_valid, _train_labels), axis=0)
    convert_images_and_labels(train_images_valid,
                              train_labels_valid,
                              os.path.join(dirpath, 'labeled_train_val.tfrecords'))
    convert_images_and_labels(unlabeled_train_images_valid,
                              unlabeled_train_labels_valid,
                              os.path.join(dirpath, 'unlabeled_train_val.tfrecords'))
    convert_images_and_labels(test_images_valid,
                              test_labels_valid,
                              os.path.join(dirpath, 'test_val.tfrecords'))


def inputs(batch_size=100,
           train=True, validation=False,
           shuffle=True, num_epochs=None):
    if validation:
        if train:
            filenames = ['labeled_train_val.tfrecords']
            num_examples = FLAGS.num_labeled_examples - FLAGS.num_valid_examples
        else:
            filenames = ['test_val.tfrecords']
            num_examples = FLAGS.num_valid_examples
    else:
        if train:
            filenames = ['labeled_train.tfrecords']
            num_examples = FLAGS.num_labeled_examples
        else:
            filenames = ['test.tfrecords']
            num_examples = NUM_EXAMPLES_TEST

    filenames = [os.path.join('seed' + str(FLAGS.dataset_seed), filename) for filename in filenames]

    filename_queue = generate_filename_queue(filenames, FLAGS.data_dir, num_epochs)
    image, label = read(filename_queue)
    image = transform(tf.cast(image, tf.float32)) if train else image
    return generate_batch([image, label], num_examples, batch_size, shuffle)


def unlabeled_inputs(batch_size=100,
                     validation=False,
                     shuffle=True):
    if validation:
        filenames = ['unlabeled_train_val.tfrecords']
        num_examples = NUM_EXAMPLES_TRAIN - FLAGS.num_valid_examples
    else:
        filenames = ['unlabeled_train.tfrecords']
        num_examples = NUM_EXAMPLES_TRAIN

    filenames = [os.path.join('seed' + str(FLAGS.dataset_seed), filename) for filename in filenames]
    filename_queue = generate_filename_queue(filenames, FLAGS.data_dir)
    image, label = read(filename_queue)
    image = transform(tf.cast(image, tf.float32))
    return generate_batch([image], num_examples, batch_size, shuffle)


def main(argv):
    prepare_dataset()


if __name__ == "__main__":
    tf.app.run()

