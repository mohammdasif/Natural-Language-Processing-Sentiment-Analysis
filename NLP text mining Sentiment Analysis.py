# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 23:07:35 2023

@author: Asif
"""

#Code: python code for extraction and to do sesntimentt analysis
	# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 09:21:30 2023

@author: Asif
"""

import requests
from bs4 import BeautifulSoup as bs # for web scraping
# Creating empty review list
sony_speaker_reviews = []
for i in range(1,30):
    speaker = []
    url = "https://www.amazon.com/Sony-SRS-XB13-Waterproof-Bluetooth-SRSXB13/dp/B08ZJ6DQNY/ref=cm_cr_arp_d_product_top?ie=UTF8&th=1"+str(i)
    header={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36'}
    response = requests.get(url,headers = header)
    # Creating soup object to iterate over the extracted content
    soup = bs(response.text,"lxml")
    # Extract the content under the specific tag
    reviews = soup.find_all("div",{"data-hook":"review-collapsed"})
    for i in range(len(reviews)):
        speaker.append(reviews[i].text)
    # Adding the reviews of one page to empty list which in future contains all the reviews
    sony_speaker_reviews += speaker
# Writing reviews in a text file
with open('sony_speaker_reviews.txt','w', encoding = 'utf8') as output:
    output.write(str(sony_speaker_reviews))
    
    import re
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download("stopwords")
from nltk.corpus import stopwords
# Joining all the reviews into a single paragraph 
sn_rev_string = " ".join(sony_speaker_reviews)
# Change to lowercase and remove unwanted symbols in case if exist
sn_rev_string = re.sub("[^A-Za-z" "]+"," ",sn_rev_string).lower()
sn_rev_string = re.sub("[0-9" "]+"," ",sn_rev_string)
# words that are contained in Sony speaker reviews
sn_reviews_words = sn_rev_string.split(" ")
# Lemmatizing
wordnet = WordNetLemmatizer()
sn_reviews_words=[wordnet.lemmatize(word) for word in sn_reviews_words]
# Filtering Stop Words
stop_words = set(stopwords.words("english"))
stop_words.update(['amazon', 'product', 'speaker', 'sony'])
sn_reviews_words = [w for w in sn_reviews_words if not w.casefold() in stop_words]


from sklearn.feature_extraction.text import TfidfVectorizer
# TFIDF: bigram
bigrams_list = list(nltk.bigrams(sn_reviews_words))
bigram = [' '.join(tup) for tup in bigrams_list]
use_idf=[]
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(bigram)
vectorizer.vocabulary_
sum_words = X.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
words_dict = dict(words_freq)
wordCloud = WordCloud(height=1400, width=1800)
wordCloud.generate_from_frequencies(words_dict)
plt.title('Most Frequently Occurring Bigrams')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
# initialize VADER
sid = SentimentIntensityAnalyzer()
pos_word_list=[]
neu_word_list=[]
neg_word_list=[]
for word in sn_reviews_words:
    if (sid.polarity_scores(word)['compound']) >= 0.25:
        pos_word_list.append(word)
    elif (sid.polarity_scores(word)['compound']) <= -0.25:
        neg_word_list.append(word)
    else:
        neu_word_list.append(word)                
# Positive word cloud
# Choosing the only words which are present in positive words
sn_pos_in_pos = " ".join ([w for w in pos_word_list])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(sn_pos_in_pos)
plt.title("Positive Words In The Review of Sony Speaker")
plt.imshow(wordcloud_pos_in_pos, interpolation="bilinear")
# negative word cloud
# Choosing the only words which are present in negwords
sn_neg_in_neg = " ".join ([w for w in neg_word_list])
wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(sn_neg_in_neg)
plt.title("Negative Words In The Review of Sony Speaker")
plt.imshow(wordcloud_neg_in_neg, interpolation="bilinear")
