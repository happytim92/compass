#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import re 
import json
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# 1. 데이터에 대한 기본 정보를 체크해보았습니다. 

# In[2]:


SM_data = pd.read_csv('./data/csv/converted_data_0924.csv')
SM_data.head()


# In[8]:


file_size = os.path.getsize('./data/csv/converted_data_0924.csv') 
print('파일 크기:', round(file_size/1000000,2), 'MB')
print('데이터 전체 개수: {}'.format(len(SM_data)))
print('데이터 전체 컬럼수: {}'.format(len(SM_data.columns)))


# ----

# 2. 데이터에 대한 전반적인 EDA를 진행하였습니다. 

# 2-1. 기본통계 

# In[10]:


basic_stats = SM_data.describe(include='all')
basic_stats


# - 총 158,119개의 데이터가 있습니다.
# - 리뷰는 대부분 쇼핑몰에서 왔으며, 화장품 카테고리가 가장 많습니다.
# - 'OO 주름관리 멀티밤 더블세트(멀티밤4+미스트2+오일2)+쇼핑백2' 제품이 가장 많은 리뷰를 받았습니다.

# 2-2 결측치 확인 

# In[12]:


missing_values = SM_data.isnull().sum()
missing_values


# - 'ProductName' 컬럼에 5개의 결측치가 있습니다.
# - 'GeneralPolarity' 컬럼에 13,711개의 결측치가 있습니다.

# 2-3 리뷰 점수 분포 

# In[13]:


plt.figure(figsize=(8, 6))
sns.countplot(data=SM_data, x='ReviewScore', palette='viridis')
plt.title('Review Score Distribution')
plt.xlabel('Review Score')
plt.ylabel('Count')
plt.tight_layout()
plt.grid(True, axis='y')
review_score_plot = plt.gcf()


# - 대부분의 리뷰 점수는 4점과 5점입니다.
# 

# 2-4 리뷰 길이 분포 

# In[17]:


plt.figure(figsize=(8, 6))
sns.histplot(data=SM_data, x='Word', bins=50, color='c')
plt.title('Review Length Distribution (by Word)')
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.tight_layout()
plt.grid(True, axis='y')
review_length_plot = plt.gcf()


# - 대부분의 리뷰는 1~20 단어 사이에 분포하고 있습니다.
# 

# 2-5 감성 극성 분포 

# In[15]:


plt.figure(figsize=(8, 6))
sns.countplot(data=SM_data, x='SentimentPolarity', hue='SentimentPolarity', palette='coolwarm')
plt.title('Sentiment Polarity Distribution')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Count')
plt.tight_layout()
plt.grid(True, axis='y')
sentiment_polarity_plot = plt.gcf()


# - 대부분의 리뷰가 긍정적입니다.
# 

# 2-6 제품별 리뷰 수 

# In[19]:


product_reviews = SM_data['ProductName'].value_counts().head(10)
product_reviews


# - 가장 많은 리뷰를 받은 제품은 'OO 주름관리 멀티밤 더블세트(멀티밤4+미스트2+오일2)+쇼핑백2'로, 1,364개의 리뷰를 받았습니다.

# ---

# 리뷰에 대한 EDA

# In[35]:


# 리뷰 통계 정보 
review_syllable_length = SM_data['Syllable']
review_word_length = SM_data['Word']

print('리뷰 길이 최댓값: {}'.format(np.max(review_syllable_length)))
print('리뷰 길이 최솟값: {}'.format(np.min(review_syllable_length)))
print('리뷰 길이 평균값: {:.2f}'.format(np.mean(review_syllable_length)))
print('리뷰 길이 표준편차: {:.2f}'.format(np.median(review_syllable_length)))
print('리뷰 길이 제1사분위: {}'.format(np.percentile(review_syllable_length,25)))
print('리뷰 길이 제3사분위: {}'.format(np.percentile(review_syllable_length,75)))
print('평균 음절 수: {}'.format(round(review_syllable_length.mean(),0)))
print('평균 단어 수: {}'.format(round(review_word_length.mean(),0)))


# 리뷰는 평균적으로 80 음절,18개 단어의 길이를 가집니다. 

# In[27]:


review_data = [review for review in SM_data['RawText']]
review_data


# In[49]:


def wordcloud(word):
    white_wordcloud = WordCloud(font_path ="AppleGothic",
                        width=480, height=480,
                        background_color='white',
                        colormap='summer').generate(' '.join(word))
    plt.imshow(white_wordcloud)
    plt.axis('off')
    plt.show()
    
wordcloud(review_data)


# In[57]:


positive_reviews = SM_data[SM_data['SentimentPolarity'] == 1]['SentimentText']
negative_reviews = SM_data[SM_data['SentimentPolarity'] == -1]['SentimentText']

wordcloud(positive_reviews),wordcloud(negative_reviews)


# In[ ]:




