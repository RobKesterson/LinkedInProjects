"""
Robert Kesterson

Sentiment analysis is the idea of reading large amounts of textual data associated with things like product reviews,
telephone communication transcripts, movie reviews, or other feedback where we want to understand public sentiment at
scale.

In this example I will train a sentiment classifier using logistic regression (regression without an l1 or l2 penalty)
for a collection 1,250 food product reviews from Amazon entitled "found_products.csv"

required packages:
matplotlib
numpy
pandas
seaborn
scikitlearn
"""

import math
import string

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

sns.set()

# read in the data
print("Loading \"food_proudcts.csv\"...")
products = pd.read_csv('food_products.csv')
print("Success \n")

# first we will strip out the products with a 3-star rating as they tend to be neutral
products = products[products['rating'] != 3]
products = products.copy()
print("Removing reviews with a 3 star rating")

# next we will apply a sentiment score of +1 to all reviews with rating > 3 and a -1 score to all reviews with rating
# < 3
print("Adding sentiment score")
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

print("Removing punctuation from reviews")
# remove all punctutation from each review
def remove_punctuation(text):
    if type(text) is str:
        return text.translate(str.maketrans('', '', string.punctuation))
    else:
        return ''

products['review_clean'] = products['review'].apply(remove_punctuation)

# next we use countvectorizer to get counts for each word, then represent each word as a new dimension
# Make counts
vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(products['review_clean'])

# Make a new DataFrame with the counts information
product_data = pd.DataFrame(count_matrix.toarray(),
        index=products.index,
        columns=vectorizer.get_feature_names_out())

# Add the old columns to our new DataFrame.
# We won't use review_clean and the summary in our model, but we will keep
# them to look at later.
product_data['sentiment'] = products['sentiment']
product_data['review_clean'] = products['review_clean']
product_data['summary'] = products['summary']

# split the data into 80% train, 10% validation, and 10% test sets
train_data, test_data = train_test_split(product_data, test_size=0.2, random_state=3)
validation_data, test_data = train_test_split(test_data, test_size=0.5, random_state=3)
print("Splitting data into 80% train, 10% validation, and 10% test sets")

# Now we will determine the majority classifier and what percentage of the reviews that have not been given 3-star
# ratings are represented by that classifier
positive_sentiment_count = 0
negative_sentiment_count = 0

train_data.reset_index()

for index, row in train_data.iterrows():
    if row['sentiment'] == 1:
        positive_sentiment_count += 1
    elif row['sentiment'] == -1:
        negative_sentiment_count += 1

majority_label = 0

if positive_sentiment_count > negative_sentiment_count:
    majority_label = 1
elif positive_sentiment_count < negative_sentiment_count:
    majority_label = -1
elif positive_sentiment_count == negative_sentiment_count:
    majority_label = 1

# majority_label
# sum(train_data['sentiment'] == 1)
majority_classifier_validation_accuracy = sum(validation_data['sentiment'] == majority_label) / len(validation_data)
print("The majority label is: " + str(majority_label) + " with accuracy of " + str(round(majority_classifier_validation_accuracy *100, 2)) + "% on the validation data set\n")

# train logistic regression classifier without penalty
features = vectorizer.get_feature_names_out()

# Note: C = 1/Lambda. Setting C to a really high value is the same as setting lambda = 0
sentiment_model = LogisticRegression(penalty='l2', random_state=1, C=1e23)
sentiment_model.fit(train_data[features], train_data['sentiment'])
print("Training logistic regression classifier (sentiment_model) without penalty on the training data")

index_counter = 0
most_negative_word = ''
most_negative_word_coefficient = 0
most_positive_word = ''
most_positive_word_coefficient = 0

for feature in sentiment_model.feature_names_in_:
    if sentiment_model.coef_[0][index_counter] > 0 and sentiment_model.coef_[0][index_counter] > most_positive_word_coefficient:
        most_positive_word = feature
        most_positive_word_coefficient = sentiment_model.coef_[0][index_counter]
    if sentiment_model.coef_[0][index_counter] < 0 and sentiment_model.coef_[0][index_counter] < most_negative_word_coefficient:
        most_negative_word = feature
        most_negative_word_coefficient = sentiment_model.coef_[0][index_counter]
    index_counter += 1
print("The most positive word in the classifier is " + str(most_positive_word) + "with a coefficient of " + str(most_positive_word_coefficient))
print("The most negative word in the classifier is " + str(most_negative_word) + "with a coefficient of " + str(most_negative_word_coefficient) + "\n")

# Next we will apply our sentiment classifier to the validation data set
print("Now using sentiment_model classifier to predict scores for the validation data set")
index_counter = 0
highest_probability_positive = 0
highest_probability_positive_index = 0
highest_probability_negative = 0
highest_probability_negative_index = 0
most_positive_review = 0
most_negative_review = 0

for predictions in sentiment_model.predict_proba(validation_data[features]):
    if predictions[0] > highest_probability_negative:
        highest_probability_negative = predictions[0]
        highest_probability_negative_index = index_counter
    if predictions[1] > highest_probability_positive:
        highest_probability_positive = predictions[1]
        highest_probability_positive_index = index_counter
    index_counter += 1

# Which reviews have the highest likelihood of being positive and negative respectively?
most_positive_review = validation_data.iloc[highest_probability_positive_index].review_clean
print("The predicted most positive review from the validation set is: \n" + str(most_positive_review))
most_negative_review = validation_data.iloc[highest_probability_negative_index].review_clean
print("The predicted most negative review from the validation set is: \n" + str(most_negative_review) + "\n")

# Now let us determine the accuracy of the sentiment classifier model on the validation data set
from sklearn.metrics import accuracy_score

sentiment_model_validation_accuracy = accuracy_score(validation_data['sentiment'], sentiment_model.predict(validation_data[features]))
print("Sentiment model is found to have " + str(round(sentiment_model_validation_accuracy*100, 2)) + "% accuracy on validation data set \n")

"""
Since our logistic regression model above can be shown to be about ~10 - 15% more accurate on average for  the 
validation data set, we would like to know a bit more about it's performance characteristics overall. To accomplish 
this we will use a confusion matrix
"""

print("Preparing confusion matrix for sentiment_model predictions on the validation data set")
def plot_confusion_matrix(tp, fp, fn, tn):
    """
    Plots a confusion matrix using the values
       tp - True Positive
       fp - False Positive
       fn - False Negative
       tn - True Negative
    """
    data = np.matrix([[tp, fp], [fn, tn]])
    sns.heatmap(data,annot=True,xticklabels=['Actual Pos', 'Actual Neg'],yticklabels=['Pred. Pos', 'Pred. Neg'])

predicted_validation_data = sentiment_model.predict(validation_data[features])
tp = sum((validation_data['sentiment'] == 1) & (predicted_validation_data == 1))
fp = sum((validation_data['sentiment'] == -1) & (predicted_validation_data == 1))
fn = sum((validation_data['sentiment'] == 1) & (predicted_validation_data == -1))
tn = sum((validation_data['sentiment'] == -1) & (predicted_validation_data == -1))

plot_confusion_matrix(tp, fp, fn, tn)
plt.show()
print("Sentiment_model has " + str(round((tp / (tp + fp)) * 100, 2)) + "% precision and " + str(round((tp / (tp + fn)) * 100, 2)) + "% recall on the validation data set")


