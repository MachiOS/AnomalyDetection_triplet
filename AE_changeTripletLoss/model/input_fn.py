"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf

import model.mnist_dataset as mnist_dataset


def filter_less_2(images, label):
    # return tf.math.greater(label,5)
    return tf.math.less(label,2)


def filter_less_3(images, label):
    # return tf.math.greater(label,5)
    return tf.math.less(label,3)

def filter_greater(images, label):
    return tf.math.greater(label,4)

def filter_equal_1(image,label):
    return tf.math.equal(label,1)

def filter_equal_0(image,label):
    return tf.math.equal(label,0)

def filter_equal_2(image,label):
    return tf.math.equal(label,2)

def train_input_fn(data_dir, params):
    """Train input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = mnist_dataset.train(data_dir)
    dataset = dataset.filter(filter_less_3) # use subset of data with specific label.

    dataset_normal_1 = dataset.filter(filter_equal_1)#
    dataset_anomal_0 = dataset.filter(filter_equal_0)
    # dataset_anomal_2 = dataset.filter(filter_equal_2)

    # dataset_anomal_2_sub = dataset_anomal_2.take(300)#reduce number 
    dataset_anomal_0_sub = dataset_anomal_0.take(137)#reduce number 
    dataset_all = dataset_normal_1.concatenate(dataset_anomal_0_sub)#concatenate all
    # dataset_all = dataset_all.concatenate(dataset_anomal_2_sub)

    dataset_all = dataset_all.shuffle(params.train_size)  # whole dataset into the buffer
    dataset_all = dataset_all.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset_all = dataset_all.batch(params.batch_size) 
    dataset_all = dataset_all.prefetch(1)  # make sure you always have one batch ready to serve

    return dataset_all


def test_input_fn(data_dir, params):
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = mnist_dataset.test(data_dir)

    dataset = dataset.filter(filter_less_3) # use subset of data with specific label.

    dataset_normal_1 = dataset.filter(filter_equal_1)

    dataset_anomal_0 = dataset.filter(filter_equal_0)
    dataset_anomal_0_sub = dataset_anomal_0.take(23)
    # dataset_all = dataset_normal_1.concatenate(dataset_anomal_0)

    dataset_all = dataset_normal_1.concatenate(dataset_anomal_0_sub)

    dataset_all = dataset_all.batch(params.batch_size)
    dataset_all = dataset_all.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset_all


def train_input_normal_all(data_dir,params):

    dataset = mnist_dataset.train(data_dir)
    dataset = dataset.filter(filter_equal_1)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)

    return dataset





