
import scipy.stats

import numpy as np
import tensorflow as tf





def mean_and_cov(embeddings):
    '''calculate mean vector and covariance matrix of the embeddings
    - Embeddings are n x d matrix'''
    
    mean = np.mean(embeddings,axis=0)
    cov = np.cov(embeddings,rowvar=0)

    return mean, cov


def multivariate_gaussian_pdf(mean,cov):
    '''Returns the denominator of pdf of a nultivariate gaussian distribution
    - X, mean are p x 1 vectors
    - cov is a p x p matrix'''
    #Initialize and reshape
   
    mean = mean.reshape(-1,1)
    p,_ = cov.shape

    #Compute values
    cov_inv = np.linalg.inv(cov)
    denominator = np.sqrt((2 * np.pi)**p * np.linalg.det(cov))
    
    return denominator
   

def calculate_probability(embedding,denominator,mean,cov):
    '''- embedding is p x 1 vectors'''

    embedding = embedding.reshape(-1,1)
    
    mean = mean.reshape(-1,1)
    cov_inv = np.linalg.inv(cov)

    exponent = -(1/2) * ((embedding - mean).T @ cov_inv @ (embedding - mean))

    return float((1. / denominator) * np.exp(exponent)) 


# def get_probabiltiy(embeddings):

#     mean,cov = mean_and_cov(embeddings)
   
#     denominator = multivariate_gaussian_pdf(mean,cov)

#     probs  = []
#     for emb in embeddings:
#         # print(emb.shape)
#         # print(emb.reshape(-1,1).shape)
#         p = calculate_probability(emb,denominator,mean,cov)
        
#         probs.append(p)
    
  
#     # return tf.convert_to_tensor(probs)
#     return probs

def get_probabiltiy(trained_embeddings,normal_embeddings,anormal_embeddings):

    mean,cov = mean_and_cov(trained_embeddings)
   
    denominator = multivariate_gaussian_pdf(mean,cov)

    normal_probs  = []
    anormal_probs = []
    train_normal_probs = []

    for emb in normal_embeddings:
        # print(emb.shape)
        # print(emb.reshape(-1,1).shape)
        p = calculate_probability(emb,denominator,mean,cov)
        normal_probs.append(p)
    

    for emb in anormal_embeddings:
       
        p = calculate_probability(emb,denominator,mean,cov)
        anormal_probs.append(p)
    

    for emb in trained_embeddings:
       
        p = calculate_probability(emb,denominator,mean,cov)
        train_normal_probs.append(p)
    
    # return tf.convert_to_tensor(probs)
    return  normal_probs,anormal_probs,train_normal_probs


def detect_anomalies(probability,params):
    '''probaibltiy is a tensor of the probaibltiy of each embeddings
    returns tensor. False is normal and True is anoamly'''

    # probability = get_probabiltiy(embeddings)

    # return tf.math.less(probability,params.threshold)
    return tf.math.less(probability,0.05)


# def show_confusionmatrix()



'''maybe in the anomaly_detection.py'''

# def print_stats(predictions,lables):
#     print("Accuracy = {}".format(accuracy_score(labels, preds)))
#     print("Precision = {}".format(precision_score(labels, preds)))
#     print("Recall = {}".format(recall_score(labels, preds)))

# def standadize(embeddings):
#     '''standadize embeddings'''
#     standadized_embeddings = scipy.stas.zscore(embeddings)

#     return standadized_embeddings


# def show_stats(embeddings):
#     probability = get_probabiltiy(embeddings)
#     probs_series = pd.Series(probs)
#     probs_series.describe()

# def plot(probaiblity:
#     probs_series = pd.Series(probaibltiy)
#     probs_series.plot.hist(bins=1000,color='red',fondsize = 40)


    


    
    

