from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import re


def readAndProcessingTweets(dataFile):
    data = pd.read_csv(dataFile, header=0)
    tweets = list(data.text)
    return tweets


class VaderSentimentAnalysis:
    testData = "test.csv"
    listOfTweets = readAndProcessingTweets(testData)
    sia = SentimentIntensityAnalyzer()

    for tweets in listOfTweets:
        sentiment_dictionary = sia.polarity_scores(tweets)
        print(sentiment_dictionary)

    print("Example Sentences")
    sentiment_dictionary = sia.polarity_scores("This place makes my taste buds want to pack up and move to a different country!")
    print(sentiment_dictionary)
    sentiment_dictionary = sia.polarity_scores("Self-driving cars? Say Hasta La Vista, Baby!")
    print(sentiment_dictionary)
