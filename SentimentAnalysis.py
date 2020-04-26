# Sentiment Analysis for NLP
# Affaan Ghazzali, David Hobby, Zain Khan

import csv
import pandas as pd
import numpy as np
import re
from string import punctuation
from collections import Counter


def readAndProcessTrainingData(dataFile, returnVal = True): # true for tweets, false for sentiment
    data = pd.read_csv(dataFile, header=0)
    tweets = list(data.text)
    tweets = [tweet.lower() for tweet in tweets]  # converts all elements of list tweets to lowercase
    tweets = [''.join(c for c in tweet if c not in punctuation) for tweet in tweets]  # removes punctuation
    sentiments = list(data.sentiment)
    for tweet in tweets:
        tweet = re.sub(r'https?\S+/', '', tweet, flags=re.MULTILINE) # removal of urls
        tweet = re.sub(r'ì\S+', '', tweet, flags=re.MULTILINE) # removal of strange utf-8 encoding
    if returnVal is False:
        return sentiments
    else:
        return tweets


def readAndProcessingTestData(dataFile):
    data = pd.read_csv(dataFile, header=0)
    tweets = list(data.text)
    tweets = [tweet.lower() for tweet in tweets]
    tweets = [''.join(c for c in tweet if c not in punctuation) for tweet in tweets]  # removes punctuation
    id = list(data.id)
    print(tweets)
    print(id)


def splitTweets(dataFile):
    data = pd.read_csv(dataFile, header=0)
    tweets = list(data.text)
    tweets = [tweet.lower() for tweet in tweets]
    tweets = [''.join(c for c in tweet if c not in punctuation) for tweet in tweets]  # removes punctuation
    for tweet in tweets:
        tweet_split = tweet.split(' ')
        print('Tweets : ', tweet_split)


class SentimentAnalysis:
    """
    Initializing Data
    """
    trainingData = "train.csv"
    testData = "test.csv"

    listOfSentiments = readAndProcessTrainingData(trainingData, False)
    listOfTweets = readAndProcessTrainingData(trainingData)

    '''
    Tokenizing Data 
    '''
    stringOfTweets = ' '.join(listOfTweets)
    words = stringOfTweets.split()
    wordscount = Counter(words)
    totalWords = len(words)
    sortedWords = wordscount.most_common(totalWords)
    global vocabToInt # solution to vocabToInt definition bug
    vocabToInt = {w: i + 1 for i, (w, c) in enumerate(sortedWords)}

    '''
    Encoding words in tweets to ints
    '''
    encodedTweets = []
    for tweet in listOfTweets:
        r = [vocabToInt[w] for w in tweet.split()]
        encodedTweets.append(r)
    #print(encodedTweets)

    '''
    Encoding sentiment labels to simplify problem
    '''
    encodedLabels = [] # for now classifying them as either positive (5,4,3) or negative (2,1)
    for sentiment in listOfSentiments:
        if sentiment >= 3:
            sentiment = 1
            encodedLabels.append(sentiment)
        else:
            sentiment = 0
            encodedLabels.append(sentiment)
    encodedLabels = np.array(encodedLabels)
    print(encodedLabels)

    '''
    TODO: 8-16
    '''

    # print(stringOfTweets)
    # print(vocabToInt)
    # print(wordscount)

    # print(listOfTweets)
    # readAndProcessTrainingData(trainingData)
    # readAndProcesingTestData(testData)
    # splitTweets(trainingData)
    # splitTweets(testData)
    # vocab2IntMap(trainingData)
