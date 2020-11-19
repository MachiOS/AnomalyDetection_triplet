#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf

from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses


# In[7]:


class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()

        self.latent_dim = 64
        self.encoder = tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(self.latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
        layers.Dense(784, activation='sigmoid'),
        layers.Reshape((28, 28))
        ])

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, encoded):
        decoded= self.decoder(encoded)
        return decoded
    
    
#     self.num_nodes_list = params.num_nodes_list
# MNIST is 28 * 28
#     self.num_nodes_list = []
    # params.num_nodes_list = [layer 1, layer 2,... middle layer]
    
#     self.bn_momentum = params.bn_momentum
#     self.activation_type =  params.activation_type
 
        

    # build encoder and decoder:
    # self.encoder = tf.keras.Sequential()
    
    # self.original_dim = 1536
    
    # for i, num_nodes in enumerate(self.num_nodes_list):
    #     self.encoder.add(tf.keras.layers.Dense(num_nodes, activation=self.activation_type))
  
    # self.decoder =  tf.keras.Sequential()
    
    # for i, c in enumerate(reversed(channels)):
    #     if i > 0:
    #         self.decoder.add(tf.keras.layers.Dense(num_nodes, activation=activation_type))
    
    
    # self.decoder.add(tf.keras.layers.Dense(self.original_dim,activation="sigmoid"))
    

   


#why decoder and encoder are separated?


# In[8]:


def build_model(is_trainig,images,params):
    """Compute outputs of the model (embeddings for triplet loss).
    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters
    Returns:
        output: (tf.Tensor) output of the model
    """
    out = images
    
    autoencoder = AnomalyDetector()
    
    encoded = autoencoder.encode(out)
    decoded = autoencoder.decode(encoded)    
    
    return encoded, decoded


# In[9]:


def model_fn(features, labels, mode, params):
    """Model function for tf.estimator
    Args:
        features: input batch of images
        labels: labels of the images
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: contains hyperparameters of the model (ex: `params.learning_rate`)
    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    images = features
    images = tf.reshape(images, [-1, params.image_size, params.image_size, 1])
    assert images.shape[1:] == [params.image_size, params.image_size, 1], "{}".format(images.shape)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.compat.v1.variable_scope('model'):
        # Compute the embeddings with the model
        embeddings, decoded = build_model(is_training, images, params)
    
    embedding_mean_norm = tf.reduce_mean(input_tensor=tf.norm(tensor=embeddings, axis=1))
    tf.compat.v1.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'embeddings': embeddings}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels = tf.cast(labels, tf.int64)

    # Define triplet loss
    if params.triplet_strategy == "batch_all":
        loss, fraction = batch_all_triplet_loss(labels, embeddings, margin=params.margin,
                                                squared=params.squared)
    elif params.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(labels, embeddings, margin=params.margin,
                                       squared=params.squared)
    else:
        raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))
    
    
    #add reconstruction error 
    mse = tf.keras.losses.MeanSquaredError()
    loss +=  mse(images, decoded)#.numpy()

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    # TODO: some other metrics like rank-1 accuracy?
    with tf.compat.v1.variable_scope("metrics"):
        eval_metric_ops = {"embedding_mean_norm": tf.compat.v1.metrics.mean(embedding_mean_norm)}

        if params.triplet_strategy == "batch_all":
            eval_metric_ops['fraction_positive_triplets'] = tf.compat.v1.metrics.mean(fraction)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


    # Summaries for training
    tf.compat.v1.summary.scalar('loss', loss)
    if params.triplet_strategy == "batch_all":
        tf.compat.v1.summary.scalar('fraction_positive_triplets', fraction)

    tf.compat.v1.summary.image('train_image', images, max_outputs=1)

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(params.learning_rate)
    global_step = tf.compat.v1.train.get_global_step()
    if params.use_batch_norm:
        # Add a dependency to update the moving mean and variance for batch normalization
        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


# In[ ]:




