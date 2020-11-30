# AnomalyDetection_triplet


models are found in experiment.


## Visualize the trained model using tensorboard

1. in an environment with Tensorflow 1.6.0 

2, 
```python
python visualize_anomalydetect.py --model_dir experiments/batch_hard_01_1N_0AN

```

![Alt text](images/confusionMatrix.png?raw=true "CM")

3. 


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

