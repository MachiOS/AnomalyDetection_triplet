# AnomalyDetection_triplet


Train a Autoencoder for anomaly detection using triplet loss.
A trained model is found in experiment.
Using MINIST data for demonstration.
digit 1 as normal and digit 0 as anomaleis.
Train with a dataset consists of 6742 Normal data and 137 anomalous data.


## Visualize the trained model using tensorboard

1. In an environment with Tensorflow 2.3.0 

```python
python visualize_anomalydetect.py --model_dir experiments/batch_hard_01_1N_0AN

```

You will see the confusion matrix of the anomaly detection. 

![Alt text](images/confusionMatrix.png?raw=true)

2. In an environment Tensorflow 1.6.0 
```bash
tensorboard --logdir experiments/batch_hard_01_1N_0AN
```

Access to 
```
http://localhost:6006/
```
Go to Projecter tub

![Alt text](images/emb.png?raw=true)

You will see the trained embeddings appears in the middle layer of the autoencoder. (projected into 3-D space)




## Train your model.

Suppose you want to train "new_model".

1. Environment: install Tensorflow tensorflow 2.3.0  

2.
```python
python train.py --model_dir experiments/new_model
```
* You need params.json (like one found in experiments/batch_hard_01_1N_0AN)inside experiments/new_model



