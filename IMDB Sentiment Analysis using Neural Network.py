#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.datasets import imdb


# In[100]:


((XT,YT),(Xt,Yt))=imdb.load_data(num_words=10000)


# In[101]:


print(XT.shape)
print(Xt.shape)
print(YT.shape)
print(Yt.shape)


# In[102]:


a=imdb.get_word_index()


# In[103]:


print(a.items())


# In[104]:


#REVERSING THE DICTIONARY
a=dict(map(reversed,a.items()))
print(a)


# In[105]:


actual_review=" ".join([a.get(idx-3,"?") for idx in XT[0]])
print(actual_review)


# In[106]:


#Vectorizing
import numpy as np
def vectorize_sentences(sentences,dim=10000):
    outputs=np.zeros((len(sentences),dim))
    for i,idx in enumerate(sentences):
        outputs[i,idx]=1
    return outputs
    


# In[107]:


X_train=vectorize_sentences(XT)
X_test=vectorize_sentences(Xt)


# In[108]:


Y_train=np.asarray(YT).astype("float32")
Y_test=np.asarray(Yt).astype("float32")
    


# In[126]:


#Defining neural network
from keras import models
from keras.layers import Dense


# In[127]:


model=models.Sequential()
model.add(Dense(16,activation="relu",input_shape=(10000,)))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation="sigmoid"))


# In[128]:


model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy"])
model.summary()


# In[129]:


#Training and Validation
X_val=X_train[:5000]
X_train_new=X_train[5000:]
Y_val=Y_train[:5000]
Y_train_new=Y_train[5000:]
b=model.fit(X_train_new,Y_train_new,epochs=4,batch_size=512,validation_data=(X_val,Y_val))


# In[130]:


hist=b.history
print(hist)


# In[131]:


import matplotlib.pyplot as plt
plt.plot(hist["val_accuracy"],label="Validation Accuracy")
plt.plot(hist["accuracy"],label="Training Accuracy")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("Accuracy")
plt.show()


# In[132]:


plt.plot(hist["val_loss"],label="Validation loss")
plt.plot(hist["loss"],label="Training loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# In[134]:


model.evaluate(X_test,Y_test)[1]


# In[135]:


model.evaluate(X_train,Y_train)[1]


# In[144]:


output=model.predict(X_test)
print(output)


# In[156]:


for i in output:
    if i>0.5:
        print("Positive Review")
    else:
        print("Negative Review")


# In[157]:


print(Y_test)


# In[ ]:





# In[ ]:




