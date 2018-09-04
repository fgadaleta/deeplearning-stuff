import tweepy

consumer_key = '1zIV7bf4ERZY1cUOA68gr7HAi'
consumer_secret = 'dqpGc4yzRpLCHQ3Kh3QFvo4L7vVzyaE4z9Vu9QsbE3mVbjV6Q4' 
access_token = '333412049-HF1xjZ42pFAX5SqkVcoRvLB7uGba4LGRa1TKlNUy'
access_token_secret = 'ul0bwY0efpJT1FcId7P4LONY2tnvWdkXgZDraG29MMxDr'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print tweet.text


