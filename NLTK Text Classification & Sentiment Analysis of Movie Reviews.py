#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ## Part 1

# In[1]:


# All imports
import pandas as pd
import nltk
import os
import sys
import random
import nltk
from nltk.corpus import stopwords


# In[2]:


dirPath = "/Users/anya/Documents/Grad_local/cis668/final_project/data/kagglemoviereviews/corpus/train.tsv"
limitStr = "20000" #change back to 20000

# convert the limit argument from a string to an int
limit = int(limitStr)

f = open('/Users/anya/Documents/Grad_local/cis668/final_project/data/kagglemoviereviews/corpus/train.tsv', 'r')
# loop over lines in the file and use the first limit of them
phrasedata = []
for line in f:
# ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the phrase and sentence ids, and keep the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])

# pick a random sample of length limit because of phrase overlapping sequences
random.shuffle(phrasedata)
phraselist = phrasedata[:limit]

print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')


# In[3]:


for phrase in phraselist[:10]:
    print (phrase)


# In[4]:


# create list of phrase documents as (list of words, label)
phrasedocs = []
# add all the phrases
for phrase in phraselist:
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocs.append((tokens, int(phrase[1])))

# print a few
for phrase in phrasedocs[:10]:
    print(phrase)


# In[5]:


# possibly filter tokens

# continue as usual to get all words and create word features

# feature sets from a feature definition function

# train classifier and show performance in cross-validation


# ## Part 2
# ### Using Unigrams

# In[6]:


# get all words from all movie_reviews and put into a frequency distribution
#   note lowercase, but no stemming or stopwords
all_words_list = [word for (sent,cat) in phrasedocs for word in sent]
all_words = nltk.FreqDist(all_words_list)
# get the 2000 most frequently appearing keywords in the corpus
word_items = all_words.most_common(2000)
word_features = [word for (word,count) in word_items]
print(word_features[:50])


# In[7]:


# define features (keywords) of a document for a BOW/unigram baseline
# each feature is 'contains(keyword)' and is true or false depending
# on whether that keyword is in the document
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    return features


# In[8]:


# get features sets for a document, including keyword features and category feature
featuresets = [(document_features(d, word_features), c) for (d, c) in phrasedocs]


# In[9]:


# the feature sets are 2000 words long so you may not want to look at one
featuresets[0]


# In[10]:


# training using naive Baysian classifier, training set is approximately 90% of data
train_set, test_set = featuresets[int((len(featuresets)*0.1)):], featuresets[:int((len(featuresets)*0.1))]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[11]:


# evaluate the accuracy of the classifier
# the accuracy result may vary since we randomized the documents
nltk.classify.accuracy(classifier, test_set)


# In[12]:


# show which features of classifier are most informative
classifier.show_most_informative_features(30)


# #### Utilizaing Cross Validation to get Precision, recall and f-measure scores

# In[63]:


## cross-validation ##
# this function takes the number of folds, the feature sets
# it iterates over the folds, using different sections for training and testing in turn
#   it prints the accuracy for each fold and the average accuracy at the end
def cross_validation_accuracy(num_folds, featuresets):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)


# In[14]:


# perform the cross-validation on the featuresets with word features and generate accuracy
#num_folds = 5
#cross_validation_accuracy(num_folds, featuresets)


# In[15]:


# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output:  prints precision, recall and F1 for each label
# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output:  prints precision, recall and F1 for each label
def eval_measures_saved(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label 
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))
        
    return precision_list, recall_list, F1_list


def eval_measures(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label 
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]),           "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))
    
    #return recall_list, precision_list, F1_list


## cross-validation ##
# this function takes the number of folds, the feature sets
# it iterates over the folds, using different sections for training and testing in turn
#   it prints the accuracy for each fold and the average accuracy at the end
def cross_validation_accuracy(num_folds, featuresets):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    precision_scoring = []
    recall_scoring = []
    F1_scoring = []
    # iterate over the folds
    for i in range(num_folds):
        #print("Fold " + str(i+1))
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
        
        
        
        goldlist = []
        predictedlist = []
        for (features, label) in test_this_round:
            goldlist.append(label)
            predictedlist.append(classifier.classify(features))
        
        
        #recall_list, precision_list, F1_list = eval_measures(goldlist, predictedlist)
        eval_measures(goldlist, predictedlist)
        precision_list, recall_list, F1_list = eval_measures_saved(goldlist, predictedlist)
        precision_scoring.append(precision_list)
        recall_scoring.append(recall_list)
        F1_scoring.append(F1_list)
        
        
    # find mean accuracy over all rounds
    print ('\nmean accuracy', sum(accuracy_list) / num_folds)
    
    
    # Find mean scores over all rounds
    labels = list(set(goldlist))

    a = precision_scoring[0]
    b = precision_scoring[1]
    c = precision_scoring[2]
    d = precision_scoring[3]
    e = precision_scoring[4]
    precision_scoring_mean = [(g + h + i + j + k) / 5 for g, h, i, j, k in zip(a, b, c, d, e)]

    a = recall_scoring[0]
    b = recall_scoring[1]
    c = recall_scoring[2]
    d = recall_scoring[3]
    e = recall_scoring[4]
    recall_scoring_mean = [(g + h + i + j + k) / 5 for g, h, i, j, k in zip(a, b, c, d, e)]

    a = F1_scoring[0]
    b = F1_scoring[1]
    c = F1_scoring[2]
    d = F1_scoring[3]
    e = F1_scoring[4]
    F1_scoring_mean = [(g + h + i + j + k) / 5 for g, h, i, j, k in zip(a, b, c, d, e)]

    # the evaluation measures in a table with one row per label
    print('\n\tMean Scores\n\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_scoring_mean[i]),           "{:10.3f}".format(recall_scoring_mean[i]), "{:10.3f}".format(F1_scoring_mean[i]))

#return recall_list, precision_list, F1_list


# In[16]:


cross_validation_accuracy(5,featuresets)


# ## Part 3
# ### Bigrams

# In[17]:


####   adding Bigram features   ####
# set up for using bigrams
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()


# In[18]:


# create the bigram finder on all the words in sequence
print(all_words_list[:50])
finder = BigramCollocationFinder.from_words(all_words_list)


# In[19]:


# define the top 500 bigrams using the chi squared measure
bigram_features = finder.nbest(bigram_measures.chi_sq, 500)
print(bigram_features[:50])


# In[20]:


# define features that include words as before 
# add the most frequent significant bigrams
# this function takes the list of words in a document as an argument and returns a feature dictionary
# it depends on the variables word_features and bigram_features
def bigram_document_features(document, word_features, bigram_features):
   document_words = set(document)
   document_bigrams = nltk.bigrams(document)
   features = {}
   for word in word_features:
       features['V_{}'.format(word)] = (word in document_words)
   for bigram in bigram_features:
       features['B_{}_{}'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)    
   return features


# In[21]:


# use this function to create feature sets for all sentences
bigram_featuresets = [(bigram_document_features(d, word_features, bigram_features), c) for (d, c) in phrasedocs]


# In[22]:


# number of features for document 0
print(len(bigram_featuresets[0][0].keys()))


# In[23]:


# features in document 0 - should be 1500 word features and 500 bigram features
print(bigram_featuresets[0][0])


# In[24]:


# training using naive Baysian classifier, training set is approximately 90% of data
train_set, test_set = bigram_featuresets[int((len(featuresets)*0.1)):], bigram_featuresets[:int((len(featuresets)*0.1))]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[25]:


cross_validation_accuracy(5,bigram_featuresets)


# ## Remove Stop Words

# In[26]:


stopwords = nltk.corpus.stopwords.words('english')
print(len(stopwords))
print(stopwords)


# In[27]:


# remove stop words from the all words list
new_all_words_list = [word for (sent,cat) in phrasedocs for word in sent if word not in stopwords]


# In[28]:


# continue to define a new all words dictionary, get the 2000 most common as new_word_features
new_all_words = nltk.FreqDist(new_all_words_list)
new_word_items = new_all_words.most_common(2000)


# In[29]:


new_word_features = [word for (word,count) in new_word_items]
print(new_word_features[:30])


# In[30]:


# now re-run one of the feature set definitions with the new_word_features instead of word_features
# get features sets for a document, including keyword features and category feature
featuresets = [(document_features(d, new_word_features), c) for (d, c) in phrasedocs]


# In[31]:


# training using naive Baysian classifier, training set is approximately 90% of data
train_set, test_set = featuresets[int((len(featuresets)*0.1)):], featuresets[:int((len(featuresets)*0.1))]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[32]:


cross_validation_accuracy(5,featuresets)


# ### Removing Negation Words 

# In[33]:


# this list of negation words includes some "approximate negators" like hardly and rarely
negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely',                  'rarely', 'seldom', 'neither', 'nor']


# In[34]:


# One strategy with negation words is to negate the word following the negation word
#   other strategies negate all words up to the next punctuation
# Strategy is to go through the document words in order adding the word features,
#   but if the word follows a negation words, change the feature to negated word
# Start the feature set with all 2000 word features and 2000 Not word features set to false
def NOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)
    return features


# In[35]:


# Create feature sets s before, using th NOT_features extraction function  
# train the classifier and test accuracy 
#define the feature sets
NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in phrasedocs]
# show the values of a couple of example features
print(NOT_featuresets[0][0]['V_NOTcare'])
print(NOT_featuresets[0][0]['V_always'])


# In[36]:


# training using naive Baysian classifier, training set is approximately 90% of data
train_set, test_set = NOT_featuresets[int((len(featuresets)*0.1)):], NOT_featuresets[:int((len(featuresets)*0.1))]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[37]:


cross_validation_accuracy(5,NOT_featuresets)


# ### Removing Punctuation

# In[38]:


punctuation_words = []

for (phrase,score) in phrasedocs:
    for word in phrase:
        if not word.isalpha() and len(word) == 1:
            punctuation_words.append(word)
punctuation_words = list(set(punctuation_words))


# In[39]:


# remove stop words from the all words list
new_all_words_list = [word for (sent,cat) in phrasedocs for word in sent if word not in punctuation_words]


# In[40]:


# continue to define a new all words dictionary, get the 2000 most common as new_word_features
new_all_words = nltk.FreqDist(new_all_words_list)
new_word_items = new_all_words.most_common(2000)


# In[41]:


new_word_features = [word for (word,count) in new_word_items]
print(new_word_features[:30])


# In[42]:


# now re-run one of the feature set definitions with the new_word_features instead of word_features
# get features sets for a document, including keyword features and category feature
featuresets = [(document_features(d, new_word_features), c) for (d, c) in phrasedocs]


# In[43]:


# training using naive Baysian classifier, training set is approximately 90% of data
train_set, test_set = featuresets[int((len(featuresets)*0.1)):], featuresets[:int((len(featuresets)*0.1))]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[44]:


cross_validation_accuracy(5,featuresets)


# ### POS Tag Features

# In[45]:


# this function takes a document list of words and returns a feature dictionary
# it runs the default pos tagger (the Stanford tagger) on the document
#   and counts 4 types of pos tags to use as features
def POS_features(document, word_features):
    document_words = set(document)
    tagged_words = nltk.pos_tag(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    numNoun = 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words:
        if tag.startswith('N'): numNoun += 1
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    features['nouns'] = numNoun
    features['verbs'] = numVerb
    features['adjectives'] = numAdj
    features['adverbs'] = numAdverb
    return features


# In[46]:


# define feature sets using this function
POS_featuresets = [(POS_features(d, word_features), c) for (d, c) in phrasedocs]
# number of features for document 0
print(len(POS_featuresets[0][0].keys()))


# In[47]:


# the first sentence
print(phrasedocs[0])
# the pos tag features for this sentence
print('num nouns', POS_featuresets[0][0]['nouns'])
print('num verbs', POS_featuresets[0][0]['verbs'])
print('num adjectives', POS_featuresets[0][0]['adjectives'])
print('num adverbs', POS_featuresets[0][0]['adverbs'])


# In[48]:


# training using naive Baysian classifier, training set is approximately 90% of data
train_set, test_set = POS_featuresets[int((len(POS_featuresets)*0.1)):], POS_featuresets[:int((len(POS_featuresets)*0.1))]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[49]:


cross_validation_accuracy(5,POS_featuresets)


# ### Using a sentiment lexicon with scores or counts: Subjectivity

# In[88]:


####   adding features   ####
# First run the program in the file Subjectivity.py to load the subjectivity lexicon
# copy and paste the definition of the readSubjectivity functions

# Module Subjectivity reads the subjectivity lexicon file from Wiebe et al
#    at http://www.cs.pitt.edu/mpqa/ (part of the Multiple Perspective QA project)
#
# This file has the format that each line is formatted as in this example for the word "abandoned"
#     type=weaksubj len=1 word1=abandoned pos1=adj stemmed1=n priorpolarity=negative
# In our data, the pos tag is ignored, so this program just takes the last one read
#     (typically the noun over the adjective)
#
# The data structure that is created is a dictionary where
#    each word is mapped to a list of 4 things:  
#        strength, which will be either 'strongsubj' or 'weaksubj'
#        posTag, either 'adj', 'verb', 'noun', 'adverb', 'anypos'
#        isStemmed, either true or false
#        polarity, either 'positive', 'negative', or 'neutral'

import nltk

# pass the absolute path of the lexicon file to this program
# example call:
# nancymacpath = 
#    "/Users/njmccrac/AAAdocs/research/subjectivitylexicon/hltemnlp05clues/subjclueslen1-HLTEMNLP05.tff"
# SL = readSubjectivity(nancymacpath)

# this function returns a dictionary where you can look up words and get back 
#     the four items of subjectivity information described above
def readSubjectivity(path):
    flexicon = open(path, 'r')
    # initialize an empty dictionary
    sldict = { }
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        # put a dictionary entry with the word as the keyword
        #     and a list of the other values
        sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict


# In[89]:


# create your own path to the subjclues file
SLpath = '/Users/anya/Documents/Grad_local/cis668/final_project/data/subjclueslen1-HLTEMNLP05.tff'


# In[90]:


# import the Subjectivity program as a module to use the function
#import Subjectivity
SL = readSubjectivity(SLpath)


# In[91]:


# how many words are in the dictionary
len(SL.keys())


# In[54]:


# look at words in the dictionary
print(SL['absolute'])
print(SL['shabby'])
# note what happens if the word is not there --> dog gave error
print(SL['happy'])


# In[55]:


# use multiple assignment to get the 4 items
strength, posTag, isStemmed, polarity = SL['absolute']
print(polarity)


# In[56]:


# define features that include word counts of subjectivity words
# negative feature will have number of weakly negative words +
#    2 * number of strongly negative words
# positive feature has similar definition
#    not counting neutral words
def SL_features(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    # count variables for the 4 classes of subjectivity
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)      
    return features


# In[57]:


SL_featuresets = [(SL_features(d, word_features, SL), c) for (d, c) in phrasedocs]


# In[58]:


# this gives the label of document 0
SL_featuresets[0][1]
# number of features for document 0
len(SL_featuresets[0][0].keys())


# In[59]:


# training using naive Baysian classifier, training set is approximately 90% of data
train_set, test_set = SL_featuresets[int((len(SL_featuresets)*0.1)):], SL_featuresets[:int((len(SL_featuresets)*0.1))]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[60]:


cross_validation_accuracy(5,SL_featuresets)


# ### LIWC Sentiment Lexicon

# In[92]:


import os
import sys

# returns two lists:  words in positive emotion class and
#		      words in negative emotion class
def read_words():
  poslist = []
  neglist = []

  flexicon = open('/Users/anya/Documents/Grad_local/cis668/final_project/data/kagglemoviereviews/SentimentLexicons/liwcdic2007.dic', encoding='latin1')
  # read all LIWC words from file
  wordlines = [line.strip() for line in flexicon]
  # each line has a word or a stem followed by * and numbers of the word classes it is in
  # word class 126 is positive emotion and 127 is negative emotion
  for line in wordlines:
    if not line == '':
      items = line.split()
      word = items[0]
      classes = items[1:]
      for c in classes:
        if c == '126':
          poslist.append( word )
        if c == '127':
          neglist.append( word )
  return (poslist, neglist)


# In[65]:


pos_list, neg_list = read_words()


# In[66]:


# test to see if a word is on the list
#   using a prefix test if the word is a stem with an *
# returns True or False
def isPresent(word, emotionlist):
  isFound = False
  # loop over all elements of list
  for emotionword in emotionlist:
    # test if a word or a stem
    if not emotionword[-1] == '*':
      # it's a word!
      # when a match is found, can quit the loop with True
      if word == emotionword:
        isFound = True
        break
    else:
      # it's a stem!
      # when a match is found, can quit the loop with True
      if word.startswith(emotionword[0:-1]):
        isFound = True
        break
  # end of loop
  return isFound


# In[67]:


isPresent('abandon',pos_list)


# In[68]:


for i in phrasedocs[:5]:
    print(i)


# In[69]:


pos_list


# In[70]:


# define features (keywords) of a document for a BOW/unigram baseline
# each feature is 'contains(keyword)' and is true or false depending
# on whether that keyword is in the document
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    return features


# In[71]:


# define features that include word counts of subjectivity words
# negative feature will have number of weakly negative words +
#    2 * number of strongly negative words
# positive feature has similar definition
#    not counting neutral words
def SL_features_LIWC(document, word_features, pos_features, neg_feautres):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    # count variables for the 4 classes of subjectivity
    Pos = 0
    Neg = 0
    
    for word in document_words:
        if word in pos_features:
            Pos += 1
            features['positivecount'] = Pos
        elif word in neg_feautres:
            Neg -= 1
            features['negativecount'] = Neg     
            
    return features


# In[72]:


SL_featuresets_LIWC = [(SL_features_LIWC(d, word_features, pos_list,neg_list), c) for (d, c) in phrasedocs]


# In[73]:


SL_featuresets_LIWC[0][0]


# In[74]:


# this gives the label of document 0
SL_featuresets_LIWC[0][1]
# number of features for document 0
len(SL_featuresets_LIWC[0][0].keys())


# In[93]:


# training using naive Baysian classifier, training set is approximately 90% of data
train_set, test_set = SL_featuresets_LIWC[int((len(
    SL_featuresets_LIWC)*0.1)):], SL_featuresets_LIWC[:int((len(SL_featuresets_LIWC)*0.1))]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[76]:


cross_validation_accuracy(5,SL_featuresets_LIWC)


# ### Removing stopwords, removing punctuation, then using sentiment lexicon with scores or counts: subjectivity 

# In[77]:


## Removing stopwords and punctuation
stopwords = nltk.corpus.stopwords.words('english')
stopwords_punctuation = stopwords + punctuation_words
print(len(stopwords_punctuation))
print(stopwords_punctuation)


# In[78]:


# remove stop words and punct from the all words list
new_all_words_list = [word for (sent,cat) in phrasedocs for word in sent if word not in stopwords_punctuation]


# In[79]:


# continue to define a new all words dictionary, get the 2000 most common as new_word_features
new_all_words = nltk.FreqDist(new_all_words_list)
new_word_items = new_all_words.most_common(2000)


# In[80]:


new_word_features = [word for (word,count) in new_word_items]
print(new_word_features[:30])


# In[81]:


## Sentiment Lexicon Addition using new words
featuresets_combined = [(SL_features(d, new_word_features, SL), c) for (d, c) in phrasedocs]


# In[82]:


# this gives the label of document 0
featuresets_combined[0][1]
# number of features for document 0
len(featuresets_combined[0][0].keys())


# In[83]:


# training using naive Baysian classifier, training set is approximately 90% of data
train_set, test_set = featuresets_combined[int((len(
    featuresets_combined)*0.1)):], featuresets_combined[:int((len(featuresets_combined)*0.1))]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[84]:


cross_validation_accuracy(5,featuresets_combined)


# ### Removing Stop Words and Punctuation

# In[85]:


featuresets_stop_punct = [(document_features(d, new_word_features), c) for (d, c) in phrasedocs]


# In[86]:


# training using naive Baysian classifier, training set is approximately 90% of data
train_set, test_set = featuresets_stop_punct[int((len(
    featuresets_stop_punct)*0.1)):], featuresets_stop_punct[:int((len(featuresets_stop_punct)*0.1))]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[87]:


cross_validation_accuracy(5,featuresets_stop_punct)


# ### AI Attempt - Failure

# In[ ]:


import time
start_time = time.time()

import sys
import numpy as np
from sklearn import datasets
import sklearn
from matplotlib import pyplot as plt
import torch
from numpy import round
import seaborn as sns
import pandas as pd
import nltk
import os
import sys
import torch
from torchtext.datasets import AG_NEWS
import random
import nltk
from nltk.corpus import stopwords
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

dirPath = "/Users/anya/Documents/Grad_local/cis668/final_project/data/kagglemoviereviews/corpus/train.tsv"
limitStr = "156000"

# convert the limit argument from a string to an int
limit = int(limitStr)

f = open('/Users/anya/Documents/Grad_local/cis668/final_project/data/kagglemoviereviews/corpus/train.tsv', 'r')
# loop over lines in the file and use the first limit of them
phrasedata = []
for line in f:
# ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the phrase and sentence ids, and keep the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])

# pick a random sample of length limit because of phrase overlapping sequences
random.shuffle(phrasedata)
phraselist = phrasedata[:limit]

print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')

#%% Sphrase list for sentiment
import nltk

# pass the absolute path of the lexicon file to this program
# example call:
# nancymacpath = 
#    "/Users/njmccrac/AAAdocs/research/subjectivitylexicon/hltemnlp05clues/subjclueslen1-HLTEMNLP05.tff"
# SL = readSubjectivity(nancymacpath)

# this function returns a dictionary where you can look up words and get back 
#     the four items of subjectivity information described above
def readSubjectivity(path):
    flexicon = open(path, 'r')
    # initialize an empty dictionary
    sldict = { }
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        # put a dictionary entry with the word as the keyword
        #     and a list of the other values
        sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict

# create your own path to the subjclues file
SLpath = '/Users/anya/Documents/Grad_local/cis668/final_project/data/subjclueslen1-HLTEMNLP05.tff'

# import the Subjectivity program as a module to use the function
#import Subjectivity
SL = readSubjectivity(SLpath)


phrasedocs = []
# add all the phrases
for phrase in phraselist:
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocs.append((tokens, int(phrase[1])))
    

  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)

sent = 'my name is matthew'
tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')
counter = Counter()
for (label, line) in train_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter, min_freq=1)

text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
s = text_pipeline(sent)


array = []
for count,value in enumerate(phraselist):
    s = text_pipeline(phraselist[count][0])
    #new_array.append(s)
    phraselist[count][0] = s



new_array = []
train_set_target = []
for a,b in phraselist:
    if len(a) == 13:
        new_array.append(a)
        train_set_target.append(b)

    
    
train_set_data = np.array(new_array)
train_set_target = np.array(train_set_target)


#train_set_data = train_set_data.reshape(-1,1)
train_set_target = train_set_target.reshape(-1,1)
train_set_target = train_set_target.astype(np.float32)


#%% Define a network
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H,H)
        self.linear3 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        h_relu = self.linear2(h_relu)
        y_pred = self.linear3(h_relu)
        return y_pred

# Training function
def train(X_train, Y_train, H, learning_rate, epochs=8000):
    model = TwoLayerNet(X_train.shape[1], H, Y_train.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        inputs = torch.from_numpy(X_train).float()
        labels = torch.from_numpy(Y_train).float()

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

    return model

def plot_results(Y_test, y_predicted):
    figx = plt.figure()
    plt.plot(range(len(Y_test)), Y_test, 'go', label='True data', alpha=0.5)
    plt.plot(range(len(Y_test)), y_predicted, 'ro', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    return figx


# K-fold cross-validation
from sklearn.model_selection import KFold
def kfold_CV(X, Y, K, H, learning_rate):
    hidden = H
    lr = learning_rate

    kf = KFold(n_splits=K, shuffle=True)
    rmse_trn_cv, rmse_tst_cv = np.empty(0), np.empty(0)
    r2_trn_cv, r2_tst_cv = np.empty(0), np.empty(0)
    i=0
    for trn_idx, tst_idx in kf.split(X):
        X_train, X_test = X[trn_idx, :], X[tst_idx, :]
        Y_train, Y_test = Y[trn_idx], Y[tst_idx]

        modelK = train(X_train=X_train, Y_train=Y_train, H=hidden, learning_rate=lr)
        with torch.no_grad():
            yhat_trn = modelK(torch.from_numpy(X_train).float()).numpy()
            yhat_tst = modelK(torch.from_numpy(X_test).float()).numpy()

        rmse_trn = np.sqrt(np.mean((yhat_trn - Y_train) ** 2))
        rmse_tst = np.sqrt(np.mean((yhat_tst - Y_test) ** 2))

        rmse_trn_cv = np.append(rmse_trn_cv, rmse_trn)
        rmse_tst_cv = np.append(rmse_tst_cv, rmse_tst)

        r2_trn = np.corrcoef(yhat_trn.squeeze(), Y_train.squeeze())[0, 1]**2
        r2_tst = np.corrcoef(yhat_tst.squeeze(), Y_test.squeeze())[0, 1]**2

        r2_trn_cv = np.append(r2_trn_cv, r2_trn)
        r2_tst_cv = np.append(r2_tst_cv, r2_tst)
        #yhat_trn = [round(num) for num in yhat_trn] #--> Just shows how to round data so it gets assigned a specific class
        fig_train = plot_results(Y_train, yhat_trn)
        i+=1
        fig_train.suptitle('Training plot fold ' + str(i))
        fig_test = plot_results(Y_test, yhat_tst)
        fig_test.suptitle('Test plot fold ' + str(i))
        
    return rmse_trn_cv.mean(), rmse_tst_cv.mean(),r2_trn_cv.mean(), r2_tst_cv.mean(), fig_train, fig_test, yhat_trn, yhat_tst, Y_train, Y_test

def kfold_CV_noplot(X, Y, K, H, learning_rate):
    hidden = H
    lr = learning_rate

    kf = KFold(n_splits=K, shuffle=True)
    rmse_trn_cv, rmse_tst_cv = np.empty(0), np.empty(0)
    r2_trn_cv, r2_tst_cv = np.empty(0), np.empty(0)

    for trn_idx, tst_idx in kf.split(X):
        X_train, X_test = X[trn_idx, :], X[tst_idx, :]
        Y_train, Y_test = Y[trn_idx], Y[tst_idx]

        modelK = train(X_train=X_train, Y_train=Y_train, H=hidden, learning_rate=lr)
        with torch.no_grad():
            yhat_trn = modelK(torch.from_numpy(X_train).float()).numpy()
            yhat_tst = modelK(torch.from_numpy(X_test).float()).numpy()

        rmse_trn = np.sqrt(np.mean((yhat_trn - Y_train) ** 2))
        rmse_tst = np.sqrt(np.mean((yhat_tst - Y_test) ** 2))

        rmse_trn_cv = np.append(rmse_trn_cv, rmse_trn)
        rmse_tst_cv = np.append(rmse_tst_cv, rmse_tst)

        r2_trn = np.corrcoef(yhat_trn.squeeze(), Y_train.squeeze())[0, 1]**2
        r2_tst = np.corrcoef(yhat_tst.squeeze(), Y_test.squeeze())[0, 1]**2

        r2_trn_cv = np.append(r2_trn_cv, r2_trn)
        r2_tst_cv = np.append(r2_tst_cv, r2_tst)
        
    return rmse_trn_cv.mean(), rmse_tst_cv.mean(),r2_trn_cv.mean(), r2_tst_cv.mean(), yhat_tst

#data = np.delete(data,1,axis=1)

H = 13
hidden = H
lr = 0.1
modelK = train(X_train=train_set_data, Y_train=train_set_target, H=hidden, learning_rate=lr)

rmse_trn_val, rmse_tst_val, r2_trn_val, r2_tst_val, fig_train, fig_test, yhat_trn,yhat_tst,Y_train,Y_test = kfold_CV(train_set_data, train_set_target, 5, H, lr)
'''
#%% Hyperparameter tuning: Grid Search
H_list = list(range(1,20))
lr_list = [1e-4, 1e-3, 1e-2, 1e-1]

rmse_trn = np.zeros((len(H_list), len(lr_list)))
rmse_tst = np.zeros_like(rmse_trn)
r2_trn = np.zeros((len(H_list), len(lr_list)))
r2_tst = np.zeros_like(rmse_trn)
for h, H in enumerate(H_list):
    for l, lr in enumerate(lr_list):
        rmse_trn_val, rmse_tst_val, r2_trn_val, r2_trn_val, yhat_tst = kfold_CV_noplot(train_set_data, train_set_target, 5, H, lr)
        rmse_trn[h, l] = rmse_trn_val
        rmse_tst[h, l] = rmse_tst_val
        r2_trn[h,l] = r2_trn_val
        r2_tst[h,l] = r2_tst_val
        
        print('H = {}, lr = {}: Training RMSE = {}, Testing RMSE = {}'.format(H, lr, rmse_trn_val, rmse_tst_val))
       # print('H = {}, lr = {}: Training R2 = {}, Testing R2 = {}'.format(H, lr, r2_trn_val, r2_tst_val))
        
i, j = np.argwhere(rmse_tst == np.min(rmse_tst))[0]
h_best, lr_best = H_list[i], lr_list[j]



print('H best = {}, lr best = {}'.format(h_best, lr_best))
'''


# In[ ]:




