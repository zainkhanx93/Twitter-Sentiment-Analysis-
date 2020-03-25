# Sentiment Analysis for NLP
# Affaan Ghazzali, David Hobby, Zain Khan

import csv
import pandas as pd
import re
from string import punctuation

def readAndProcessTrainingData(dataFile):
    data = pd.read_csv(dataFile, header=0)
    tweets = list(data.text)
    tweets = [tweet.lower() for tweet in tweets] # converts all elements of list tweets to lowercase
    tweets = [''.join(c for c in tweet if c not in punctuation) for tweet in tweets] # removes punctuation
    sentiments = list(data.sentiment)

    # for tweet in tweets:
    #    re.sub(r"https?\S+", "", tweet)

    print(tweets)
    print(sentiments)


def readAndProcesingTestData(dataFile):
    data = pd.read_csv(dataFile, header=0)
    tweets = list(data.text)
    tweets = [tweet.lower() for tweet in tweets]
    tweets = [''.join(c for c in tweet if c not in punctuation) for tweet in tweets] # removes punctuation
    id = list(data.id)
    print(tweets)
    print(id)


class SentimentAnalysis:
    trainingData = "train.csv"
    testData = "test.csv"
    readAndProcessTrainingData(trainingData)
    readAndProcesingTestData(testData)
