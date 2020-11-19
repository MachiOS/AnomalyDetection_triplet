#!/usr/bin/env python
# coding: utf-8

# In[12]:


num_channels = 40
channels = [num_channels, num_channels * 2]
    
# for i, c in enumerate(channels):
#     print(i)
#     print(c)


# In[18]:


for i, c in enumerate(reversed(channels)):
    if i > 0:
        print(i)
        print(c)


# In[1]:


import tensorflow as tf


# In[2]:


tf.compat.v1.reset_default_graph()


# In[3]:


mnist = tf.keras.datasets.mnist


# In[ ]:




