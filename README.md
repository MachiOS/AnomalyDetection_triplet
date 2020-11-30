# AnomalyDetection_triplet


A trained model is found in experiment.
Using MINIST data for demonstration.
digit 1 as normal and digit 0 as anomaleis.
Train with a dataset consists of 6742 Normal data and 137 anomalous data.



## Visualize the trained model using tensorboard

1. in an environment with Tensorflow 1.6.0 

2.
```python
python visualize_anomalydetect.py --model_dir experiments/batch_hard_01_1N_0AN

```

![Alt text](images/confusionMatrix.png?raw=true)

3. In an environment Tensorflow 1.6.0 
```python
tensorboard --logdir experiments/new_model
```


## Training

Suppose you want to train "new_model".

1 Environment: install Tensorflow tensorflow 2.3.0  

2. 
```python
python train.py --model_dir experiments/new_model
```
* You need params.json (like one found in experiments/batch_hard_01_1N_0AN)inside experiments/new_model


3. Visualization
```python
python visualize_anomalydetect.py --model_dir experiments/new_model

```
4. In an environment Tensorflow 1.6.0 
```python
tensorboard --logdir experiments/new_model
```

