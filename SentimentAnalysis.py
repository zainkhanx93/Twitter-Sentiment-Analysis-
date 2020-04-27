# Sentiment Analysis for NLP
# Affaan Ghazzali, David Hobby, Zain Khan

import csv
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from string import punctuation
from collections import Counter
from SentimentAnalysisLSTM import SentimentAnalysisLSTM

# MUST DO'S
    # TODO: Split training data into training/validation 90/10
    # TODO: Fix neuralNetwork training loop, probably due to lack of validation data
    # TODO: Start and finish testing loop

# IF POSSIBLE
    # TODO: if possible revert back to 1-5 sentiment instead of positive- negative



def readAndProcessTrainingData(dataFile, returnVal=True):  # true for tweets, false for sentiment
    data = pd.read_csv(dataFile, header=0)
    tweets = list(data.text)
    tweets = [tweet.lower() for tweet in tweets]  # converts all elements of list tweets to lowercase
    tweets = [''.join(c for c in tweet if c not in punctuation) for tweet in tweets]  # removes punctuation
    sentiments = list(data.sentiment)
    for tweet in tweets:
        tweet = re.sub(r'https?\S+', '', tweet, flags=re.MULTILINE)  # removal of urls
        tweet = re.sub(r'ì\S+', '', tweet, flags=re.MULTILINE)  # removal of strange utf-8 encoding
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
    return tweets
    # print(tweets)
    # print(id)


def splitTweets(dataFile):
    data = pd.read_csv(dataFile, header=0)
    tweets = list(data.text)
    tweets = [tweet.lower() for tweet in tweets]
    tweets = [''.join(c for c in tweet if c not in punctuation) for tweet in tweets]  # removes punctuation
    for tweet in tweets:
        tweet_split = tweet.split(' ')
        print('Tweets : ', tweet_split)


def tweetAnalysis(listOfEncodedTweets):  # utility function
    tweetLength = [len(tweet) for tweet in listOfEncodedTweets]
    pd.Series(tweetLength).hist()
    plt.show()

    print(pd.Series(tweetLength).describe())


def padTweets(encodedTweets, sequenceLength):  # padding each review to a sequence length of 30
    features = np.zeros((len(encodedTweets), sequenceLength), dtype=int)
    for i, tweet in enumerate(encodedTweets):
        tweetLength = len(tweet)

        if tweetLength <= sequenceLength:
            zeroes = list(np.zeros(sequenceLength - tweetLength))
            modifiedTweet = zeroes + tweet

        elif tweetLength > sequenceLength:
            modifiedTweet = tweet[0:sequenceLength]

        features[i, :] = np.array(modifiedTweet)

    return features


class SentimentAnalysis:
    """
    Initializing Data
    """
    trainingData = "train.csv"
    testData = "test.csv"

    # Training Data
    listOfTweets = readAndProcessTrainingData(trainingData)
    listOfSentiments = readAndProcessTrainingData(trainingData, False)

    # Test Data
    listOfTestTweets = readAndProcessingTestData(testData)

    '''
    Tokenizing Data 
    '''

    # Training Data
    stringOfTweets = ' '.join(listOfTweets)
    words = stringOfTweets.split()
    wordscount = Counter(words)
    totalWords = len(words)
    sortedWords = wordscount.most_common(totalWords)
    global vocabToInt  # solution to vocabToInt definition bug
    vocabToInt = {w: i + 1 for i, (w, c) in enumerate(sortedWords)}

    # Test Data
    stringOfTestTweets = ' '.join(listOfTestTweets)
    wordsTest = stringOfTestTweets.split()
    wordsTestCount = Counter(wordsTest)
    totalWordsTest = len(words)
    sortedWordsTest = wordsTestCount.most_common(totalWordsTest)
    global vocabToIntTest
    vocabToIntTest = {w: i + 1 for i, (w, c) in enumerate(sortedWordsTest)}

    '''
    Encoding words in tweets to ints
    '''

    # Training Data
    encodedTweets = []
    for tweet in listOfTweets:
        translator = [vocabToInt[w] for w in tweet.split()]  # change r to translator or something more adept
        encodedTweets.append(translator)
    # print(encodedTweets)

    # Test Data
    encodedTestTweets = []
    for testTweet in listOfTestTweets:
        translatorTest = [vocabToIntTest[w] for w in testTweet.split()]
        encodedTestTweets.append(translatorTest)

    '''
    Encoding sentiment labels to simplify problem
    '''

    # Training Sentiments
    encodedSentiments = []  # for now classifying them as either positive (5,4,3) or negative (2,1)
    for sentiment in listOfSentiments:
        if sentiment >= 3:
            sentiment = 1  # positive
            encodedSentiments.append(sentiment)
        else:
            sentiment = 0  # negative
            encodedSentiments.append(sentiment)
    encodedSentiments = np.array(encodedSentiments)
    # print(encodedLabels)

    # Testing Result Array
    testSentiments = [0] * len(encodedTestTweets)

    '''
    Analyze Tweet Length/ Padding and Truncating
    '''

    # tweetAnalysis(encodedTweets)

    # Training features
    trainingFeatures = padTweets(encodedTweets, 30)

    # Test features
    testFeatures = padTweets(encodedTestTweets, 30)

    '''
    Training and Test Data Split
    '''

    trainingX = trainingFeatures[0: int(1 * len(trainingFeatures))]
    trainingY = encodedSentiments[0: int(1 * (len(encodedSentiments)))]

    testX = testFeatures[0: int(1 * len(testFeatures))]
    testY = testSentiments[0: int(1 * len(testSentiments))]

    '''
    Dataloaders Creation
    '''

    # Tensor Data Sets
    processedTrainingTweets = TensorDataset(torch.from_numpy(trainingX), torch.from_numpy(trainingY))
    processedTestingTweets = TensorDataset(torch.from_numpy(trainingX), torch.from_numpy(trainingY))

    # Dataloaders
    batchSize = 5  # TODO: completely random guess, will need to test which batch size gets best results

    trainingLoader = DataLoader(processedTrainingTweets, shuffle=True, batch_size=batchSize)
    testingLoader = DataLoader(processedTestingTweets, shuffle=True, batch_size=batchSize)

    '''
    # Data Iterator (for visualization)
    dataIterator = iter(trainingLoader)
    sampleX, sampleY = dataIterator.next()

    print('Sample input size: ', sampleX.size()) # should be batchSize and sequenceLength
    print('Sample Input: \n', sampleX)
    print()
    print('Sample label size: ', sampleY.size()) # should be just batchSize
    print('Sample Label: \n', sampleY)
    '''

    '''
    Init LSTM RNN
    '''

    vocabSize = len(vocabToInt) + 1  # account for padding
    outputSize = 1
    embeddingDim = 400
    hiddenDim = 256
    noOfLayers = 2

    neuralNet = SentimentAnalysisLSTM(vocabSize, outputSize, embeddingDim, hiddenDim, noOfLayers)

    print(neuralNet)

    '''
    Training Loop (pretty standard training code found in many implementations of PyTorch frameworks)
    '''
    '''
    # optimization and loss functions
    learningRate = 0.001

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(neuralNet.parameters())

    # training parameters
    epochs = 3 # TODO: figure out where this is the most optimal
    counter = 0
    printer = 100
    clip = 5 # gradient

    neuralNet.train()
    # train for n of epochs
    for e in range(epochs):
        # init hidden state
        hiddenState = neuralNet.initHidden(batchSize)
        # batch loop
        for tweets, sentiments in trainingLoader:
            counter += 1

            # new variable needed for the hidden state,
            # otherwise it would go through the entire training history
            hiddenState = tuple([each.data for each in hiddenState])

            # gradient accumulated zeros
            neuralNet.zero_grad()

            # OUTPUT
            tweets = tweets.type(torch.LongTensor)
            output, hiddenState = neuralNet(tweets, hiddenState)

            # loss calculation and backprop
            losses = criterion(output.squeeze(), sentiments.float())
            losses.backward()

            # line below prevents exploding gradient problem in LSTM RNNs
            nn.util.clips_grad_norm_(neuralNet.parameters(), clip)
            optimizer.step()

            neuralNet.train()
            print("Epoch: {}?{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(losses.item()))
    '''




