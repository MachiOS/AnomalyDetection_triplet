"""Train the model"""

import argparse
import os
import pathlib
import shutil

import numpy as np
import tensorflow as tf
# from tensorflow.contrib.tensorboard.plugins import projector
from tensorboard.plugins import projector


import model.mnist_dataset as mnist_dataset
from model.utils import Params
from model.input_fn import test_input_fn
from model.model_fn import model_fn
from deploy_ad import get_probabiltiy
from deploy_ad import detect_anomalies

import pandas as pd
import matplotlib.pyplot as plt
from itertools import compress
from model.input_fn import train_input_fn
from model.input_fn import train_input_normal_all



tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/mnist',
                    help="Directory containing the dataset")
parser.add_argument('--sprite_filename', default='experiments/mnist_10k_sprite.png',
                    help="Sprite image for the projector")




if __name__ == '__main__':

    def filter_less_2(images, label):
    # return tf.math.greater(label,5)
        return tf.math.less(label,2)


    def filter_less_3(images, label):
    # return tf.math.greater(label,5)
        return tf.math.less(label,3)
    
    def filter_equal_1(image,label):
        return tf.math.equal(label,1)
    
    def filter_equal_0(image,label):
        return tf.math.equal(label,0)
    
    # @tf.function
    # def print_dataset(dataset):
    #     # for images, label in dataset:
    #     #     print(images)
    #     #     print(label)
    #     for element in dataset:
    #         print(element)

    

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Define the model
    tf.compat.v1.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)


    # EMBEDDINGS VISUALIZATION

    # Compute embeddings on the test set
    tf.compat.v1.logging.info("Predicting")
    

  

    predictions = estimator.predict(lambda: test_input_fn(args.data_dir, params))

    # print(predictions.shape)
    # TODO (@omoindrot): remove the hard-coded 10000
    embeddings = np.zeros((params.eval_size, params.embedding_size))
    
    for i, p in enumerate(predictions):
        embeddings[i] = p['embeddings']
        # print(i)

    tf.compat.v1.logging.info("Embeddings shape: {}".format(embeddings.shape))


    # Visualize test embeddings
    embedding_var = tf.Variable(embeddings, name='mnist_embedding')

    eval_dir = os.path.join(args.model_dir, "eval")

    
    summary_writer = tf.compat.v1.summary.FileWriter(eval_dir)
    # summary_writer = tf.summary.create_file_writer(eval_dir)
    
   
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()

    embedding.tensor_name = embedding_var.name

    # Specify where you find the sprite (we will create this later)
    # Copy the embedding sprite image to the eval directory
    # shutil.copy2(args.sprite_filename, eval_dir)
    # embedding.sprite.image_path = pathlib.Path(args.sprite_filename).name
    # embedding.sprite.single_image_dim.extend([28, 28])

    with tf.compat.v1.Session() as sess:
        # TODO (@omoindrot): remove the hard-coded 10000
        # Obtain the test labels
        dataset = mnist_dataset.test(args.data_dir)
    
        dataset_normal_1 = dataset.filter(filter_equal_1)
        dataset_anomal_0 = dataset.filter(filter_equal_0)

        dataset_anomal_0_sub = dataset_anomal_0.take(23)

        dataset_all = dataset_normal_1.concatenate(dataset_anomal_0_sub)
        # dataset_all = dataset_normal_1.concatenate(dataset_anomal_0)

        # dataset_all = dataset.filter(filter_less_3) # use subset of data with specific label.

        dataset_all = dataset_all.map(lambda img, lab: lab)
        dataset_all = dataset_all.batch(params.eval_size)
        labels_tensor = tf.compat.v1.data.make_one_shot_iterator(dataset_all).get_next()
        labels = sess.run(labels_tensor)


    print(labels)
    print(labels.shape)


    '''
    (BEGIN) Trained Embeddings
    '''
    # # Embeddings from training examples.
    
    trained_hidden = estimator.predict(lambda: train_input_normal_all(args.data_dir, params))

    # TODO train size
    trained_embeddings= np.zeros((6742, params.embedding_size))
    
    for i, p in enumerate(trained_hidden):
        trained_embeddings[i] = p['embeddings']
        # print(i)



    '''
    (END)Trained Embeddings
    '''

    '''
    (BEGIN) Added for analysis of the pdf
    '''
    # print(labels.shape)
    # print(labels)
    # print(type(labels))
    # print(type(embeddings))

    # print(len(labels))

    # Create Labels for anomaly detection 
    test_labels = np.array([True for i in range(len(labels))])

    # Get Labels as Boolen (True(digit 1)for normal data, False (digit 0 or 2 )for anomal data )  
    for index,label in enumerate(labels):
        if label == 1:
            test_labels[index]=True
        else:
            test_labels[index]=False

    # Get subset of embeddings (normal and anomal)
    test_labels = test_labels.astype(bool)
    normal_test_embeddings = embeddings[test_labels]
    anormal_test_embeddings = embeddings[~test_labels]


    # get Probaibltiy statistics 
    probability_normal,probability_anormal,probability_train_normal = get_probabiltiy(trained_embeddings, normal_test_embeddings,anormal_test_embeddings)
    anomalies = detect_anomalies(probability_normal,params)




    normal_probs_series = pd.Series(probability_normal)
    anormal_probs_series = pd.Series(probability_anormal)
    train_normal_probs_series = pd.Series(probability_train_normal)

    print("Normal data Probability Stats")
    print(normal_probs_series.describe())
    
    print("Anormal data Probability Stats")
    print(anormal_probs_series.describe())

    print("Train Normal data Probability Stats")
    print(train_normal_probs_series.describe())

    # Show statictics of the probabiltiy
    # plt.hist(probability_normal,bins=1000,lab el="normal")
    # plt.hist(probability_anormal,bins=1000,label="anormal")
    # plt.legend()
    # plt.show()
    
    # threshold = np.mean(probabiltiy_train_normal) + np.std(probabiltiy_train_normal)
    threshold = np.percentile(probability_train_normal,5)

    print('threshold :{0}'.format(threshold))

    misprediction = list(compress(probability_normal,np.less(probability_normal,threshold)))
    correctPrediction =  list(compress(probability_anormal,np.less(probability_anormal,threshold)))
  
    print("confusion matrix")
    TP=len(probability_normal)-len(misprediction)
    TN=len(correctPrediction)
    FP=len(probability_anormal)-len(correctPrediction)
    FN=len(misprediction)
    total_normal = len(probability_normal)
    total_anormal = len(probability_anormal)
    df_results = pd.DataFrame([[TP,FN,total_normal],[FP,TN,total_anormal]],columns=["Prd_Normal","Prd_Anormal","total"],index=["GT_Normal","GT_Anormal"])
    
    print(df_results)


    '''
    (END) Added for analysis of the pdf
    '''

    # Specify where you find the metadata
    # Save the metadata file needed for Tensorboard projector
    metadata_filename = "mnist_metadata.tsv"
    with open(os.path.join(eval_dir, metadata_filename), 'w') as f:
        for i in range(params.eval_size):
            c = labels[i]
            f.write('{}\n'.format(c))
    embedding.metadata_path = metadata_filename

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(embedding_var.initializer)
        saver.save(sess, os.path.join(eval_dir, "embeddings.ckpt"))


    


    # #Plot confusion matrix
    # confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands, 
    #             annot=True, fmt='g')
    # plt.xlabel('Prediction')
    # plt.ylabel('Label')
    # plt.show()


