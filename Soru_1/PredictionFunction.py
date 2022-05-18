#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
import pickle


# In[5]:


model = load_model('model_clean_lstm_50len-004-0.540298.h5')


# In[11]:


STOPWORDS_DIR = './merged_stopwords_tr_extended.txt'

with open(STOPWORDS_DIR, 'r') as f:
    tr_stop_words = [line.strip() for line in f]

punc_pool = string.punctuation
punc_pool = punc_pool + '’'
tags = ['economics','health','life','sports','technology']

def clean_text(text):
    
    # Para birimleri handling
    money_sign_pool = ['$', '€', '£', '₺', 'tl']
    text = "".join([' parabirimi ' if ltr in money_sign_pool else ltr for ltr in text])
 #   print(text) 
    
    # Remove Punctuations ? remove : count
    re_punct = re.compile('[%s]' % re.escape(punc_pool))
    text = re_punct.sub(' ', text)
    
    # Tokenize et
    tokens = text.split(' ')
    
    # Capital İ handling
    tokens = [re.sub("İ","i",token) for token in tokens]
    
    #Digit handling
    
    tokens = [re.sub(r'\b\d+\b', 'numeric', token.lower()) for token in tokens]
    
    # Remove Stopwords
    tokens = [w for w in tokens if w not in tr_stop_words]
 #   print(tokens)

    tokens = [token for token in tokens if len(token) > 2]
    
    pattern = re.compile(r'\s+')
    sentence = re.sub(pattern, ' ', ' '.join(tokens))
    
    return sentence


# In[16]:


filename = "soru1_vectorizer.pickle"

loaded_vectorizer = pickle.load(open(filename, 'rb'))


# In[77]:

while True:

#text = "Sağlıklı beslenmek; hastalıklardan korunmanın yanı sıra iyileşme döneminde de vücudun savunma mekanizmalarını güçlendiriyor. Tedavisinde yan etkilerle karşılaşılan kanserle de doğru beslenme desteği sayesinde daha kolay mücadele edilebiliyor. “Hayatımızın her döneminde olduğu gibi kanser tedavisi görülen dönemde de yeterli ve dengeli bir beslenme programı uygulamak önemli” diyen Dyt. Ayşe Korkmaz; kanser hastalarına, tedavi döneminde uygulayabilecekleri beslenme önerilerinde bulundu:"

    text = input('Haber metni:')


# In[78]:

    if text == ('q'):
        break
    else:

        text_cleaned = clean_text(text)


# In[79]:


        transformed = loaded_vectorizer.texts_to_sequences([text_cleaned])


# In[80]:


        padded = pad_sequences(transformed, maxlen=50, padding='pre')


# In[81]:


        yhat = model.predict(padded)


# In[82]:


        
        indice = np.argmax(yhat, axis=1)


        print("\nPredicted Label : ", tags[int(indice)])

