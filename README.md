# SentimentAnalysisProject
## NLT Project
  ### Article using RNN (recurrent nueral networks)
    https://towardsdatascience.com/sentiment-analysis-using-lstm-step-by-step-50d074f09948



## Textblob
What I learned about textblob is that it's calculate the sentiment using the polarity and subjectivity. These are built in functions in textblob. If the polarity of the tweet is greater than 0 then the tweet is positive. if it's less than 0 then the tweet is negative. If the polarity is 0 then it's neutral. 

Subjectivity is a float between 0 and 1. That calculates if the tweet is objective or subjective. If the subjectivity is 0 then then that tweet is subjective, meaning that tweet is more based of opinion then anything else. While objective means the tweet has little to no opinion. 


While doing research on textblob I discovered that it didn't use any training data like our original approach and Vader. It doesn't call any library such as dictionary. I couldn't find how its  calculate the polarity score or subjectivity. 

The most common way I saw  textblob is used in the many examples I saw wouldn't work for our project. 

The most common way I found people used textblob was to :

When doing sentiment analysis with textblob you need to connect to Twitter API and with tweepy. You access Twitter by logging in and getting some accesses keys that tweepy will use to login as you  You then have to login into the Twitter and extract tweets with a keywords like "driverless cars". Once that keyword is searched you run the sentiment function on it. The sentiment function is what gives it's score  of polarity and subjectivity. 


This method wouldn't work for our project since we are not accessing Twitter to get our tweets about driverless cars. Since we have them in a CSV file.  

The other few examples I saw on the usage of text blob didn't work either. Textblob was causing me to get some import errors. Even after installing all the plug ins and everything it still didn't work. 

We were not able to get it to work.
