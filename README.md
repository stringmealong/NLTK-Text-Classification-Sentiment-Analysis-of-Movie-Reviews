# NLTK-Text-Classification-Sentiment-Analysis-of-Movie-Reviews
NLTK Text Classification &amp; Sentiment Analysis of Movie Reviews data, code, and report
This project, completed while taking Natural Language Processing
(CIS 668) under the supervision of Dr. Michael Larche, aimed to
use the Natural Language Processing Toolkit (NLTK) package to
improve text classification models performing sentiment analysis
of movie reviews. Explicitly, the following steps were tested and
assessed via precision, recall and F-measure scores:
• filtering by stopwords or other pre-processing methods
• representing negation
• using the Subjectivity sentiment lexicon (with scores and
counts)
• using varying sizes of vocabularies
• incorporating POS tag features
• Using the LIWC sentiment lexicon
The dataset chosen was produced for the Kaggle competition,
described here: https://www.kaggle.com/c/sentiment-analysison-
movie-reviews, and which uses data from the sentiment
analysis by Socher et al, detailed at this web site:
http://nlp.stanford.edu/sentiment/. The data was taken from the
original Pang and Lee movie review corpus based on reviews from
the Rotten Tomatoes web site. Socher’s group used crowdsourcing
to manually annotate all the subphrases of sentences
with a sentiment label ranging over: “negative”, “somewhat
negative”, “neutral”, “somewhat positive”, “positive”.
This application required feature engineering using Unigrams, with
lowercase letters enforced. This allowed for features, or
keywords, to be extracted for analysis. In order to attain a
baseline evaluation of the text, this feature set was evaluated
using a Naive Bayes Classifier for Accuracy, Precision, Recall, and
F-Measure Scores.
This project satisfies the following M.S. in Applied Data Science
program's learning objectives: (2) collect and organize data, (3)
identify patterns in data via visualization, statistical analysis, and
data mining, & (4) develop alternative strategies based on the
data. To illustrate...
This project first and foremost demonstrates how data scientists
must experiment with models to attain optimal results.
Specifically, different tools within the NLTK package can be
utilized and eventually lead to various actionable insights. Using
NLTK to analyse, preprocess, and understand the written text is
an important step that can not only save time, but aid in empirical
linguistics, cognitive science, artificial intelligence, information
retrieval, and machine learning.
Furthermore, text data is incredibly important to commercial
analytics teams as more unstructured sources are introduced.
With more content being created by customers and clients such as
reviews, social media posts, and transcriptions, the value of
extracting quantifiable and actionable insights from text is
growing in significance. As organizations project a greater social
media presence, the ability to organize and analyze large
collections of text allows for automation using conversational
assistants, as well as predictive analytics with text mining.
This project also illustrates the value of analyzing all the different
parts of the metrics rather than just the accuracy-- precision,
recall, and F1-scores are also important! Holistically
understanding what information these metrics convey and the
story they tell in regards to each individual project is crucial in the
field of data science. Numbers alone do not signify anything at all,
rather, the conceptual understanding behind them is the
powerhouse in making data-driven decisions.
Thus, the project could be improved by heeding a holistic
interpretation of the metrics and progressing to reprocessing the
text, thereby improving the performance of the Naive Bayes
classifier.
