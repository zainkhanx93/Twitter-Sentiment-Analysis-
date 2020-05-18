from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import re


def readAndProcessingTweets(dataFile):
    data = pd.read_csv(dataFile, header=0)
    tweets = list(data.text)
    return tweets


class VaderSentimentAnalysis:
    testData = "test.csv"
    #listOfTweets = readAndProcessingTweets(testData)
    sia = SentimentIntensityAnalyzer()

    #for tweets in listOfTweets:
    sentiment_dictionary = sia.polarity_scores("Two places I'd invest all my money if I could: 3D printing and Self-driving cars!!!")
    print(sentiment_dictionary)
    sentiment_dictionary = sia.polarity_scores("Awesome! Google driverless cars will help the blind travel more often; https://t.co/QWuXR0FrBpv")
    print(sentiment_dictionary)
    sentiment_dictionary = sia.polarity_scores("Autonomous vehicles could reduce traffic fatalities by 90%...I'm in!")
    print(sentiment_dictionary)