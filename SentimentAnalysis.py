# Sentiment Analysis for NLP
# Affaan Ghazzali, David Hobby, Zain Khan

import csv
import pandas as pd
import numpy as np
import re
from string import punctuation
from collections import Counter

def readAndProcessTrainingData(dataFile):
    data = pd.read_csv(dataFile, header=0)
    tweets = list(data.text)
    tweets = [tweet.lower() for tweet in tweets] # converts all elements of list tweets to lowercase
    tweets = [''.join(c for c in tweet if c not in punctuation) for tweet in tweets] # removes punctuation
    sentiments = list(data.sentiment)

    # for tweet in tweets:
    #    re.sub(r"https?\S+", "", tweet)

   # print(tweets)
    #print(sentiments)


def readAndProcesingTestData(dataFile):
    data = pd.read_csv(dataFile, header=0)
    tweets = list(data.text)
    tweets = [tweet.lower() for tweet in tweets]
    tweets = [''.join(c for c in tweet if c not in punctuation) for tweet in tweets] # removes punctuation
    id = list(data.id)
   # print(tweets)
   # print(id)

def spiltTweets(dataFile):
    data = pd.read_csv(dataFile, header=0)
    tweets = list(data.text)
    tweets = [tweet.lower() for tweet in tweets]
    tweets = [''.join(c for c in tweet if c not in punctuation) for tweet in tweets]  # removes punctuation
    for tweet in tweets:
        tweet_split = tweet.split(' ')
        print('Tweets : ',  tweet_split)



class SentimentAnalysis:
    trainingData = "train.csv"
    testData = "test.csv"
    readAndProcessTrainingData(trainingData)
    readAndProcesingTestData(testData)
    spiltTweets(testData)
