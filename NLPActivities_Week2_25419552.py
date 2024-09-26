"""
Course: Skills Bootcamp in Cloud Computing and Data Analytics
Student Name: Caroline Lau Campbell
Student ID: 25419552

Assignment: IOC6 Natural Language Processing (Tutorial 2)

NB: This program is an adaptation of my coursework for a module I studied on 
my Edgehill course (2022). Uses the built-in NLTK corpora and trained models 
available at <https://www.nltk.org/nltk_data/>.
"""

### Import statements ###
import pandas as pd
from textblob import TextBlob
import random
from nltk import word_tokenize, pos_tag, ne_chunk # For NER.
from nltk.tree import Tree # For NER.

### Load dataset ###
df = pd.read_csv('Video_games_reviews.csv', delimiter='\t', header=None)
# Specify tab delimiter and no headers.
pd.set_option('display.width', 1200)
pd.set_option('display.max_columns', 100)
# NB: Video games dataset contains 20 reviews.
video_review_texts = df[2] # Reviews are in the third column.

print()
print(f'*** Video game reviews dataset ***\n')
print(df.head(3)) # First 3 rows of dataset.
print(f'...')
print(df.tail(3)) # Last 3 rows of dataset.
print(f'\n')

### Random number generator for sample output ###
def random_3_num_list(max_num):
    """Helper function to generate 3 random numbers.

    Args:
        max_num (int): Maximum number of items.

    Returns:
        list: Three random numbers to be used for iteration.
    """    
    # NB: Ideally, max number should be >=3
    sample_size = min(max_num, 3) # Adjust sample size if max is <3
    return random.sample(range(max_num), sample_size)

### Analysis of review sentiments ###
print(f'*** Sample output of review sentiment analysis ***\n')
random_review_indices = random_3_num_list(len(video_review_texts))
# Select 3 random reviews from dataset.
for review_index in random_review_indices:
    review_text = video_review_texts[review_index]
    blob = TextBlob(review_text) # Call text blob for current review_text.
    print(f'-------Analysing review #{review_index + 1}-------')
    print(f'{review_text}')
    print(f'-------END-------\n')   
    sentences = blob.sentences # List of review sentences.
    # Output sentence sentiment scores for 3 of the sentences.
    random_sentence_indices = random_3_num_list(len(sentences))
    for sentence_index in random_sentence_indices:
        sentence = sentences[sentence_index]
        print(f'-------SENTIMENT OF SENTENCE #{sentence_index + 1}-------')
        print(f'{sentence} \t {sentence.sentiment.polarity}')
        print(f'-------END-------\n')
    print()

### Set classification threshold to average of all review polarities ###
sentiment_classification_labels = []
average_review_polarity = 0
i = 0
for index, review_text in enumerate(video_review_texts):
    i += 1
    blob = TextBlob(review_text)
    review_polarity = 0
    for sentence in blob.sentences:
        review_polarity += sentence.sentiment.polarity
    if review_polarity > 0.27:
        sentiment_label_for_current_review = 1 # positive
    else:
        sentiment_label_for_current_review = 0 # negative
    sentiment_classification_labels.append(sentiment_label_for_current_review)
    average_review_polarity += review_polarity

print(f'Average sentiment polarity per review: {average_review_polarity/i}\n')
# 0.27599825793901306
print(f'')

df['Sentiment_Classification_Labels'] = sentiment_classification_labels 
# Append labels to df.

### Results of analysis ###
print(f'*** Negative video game reviews ***\n')
print(df[df.Sentiment_Classification_Labels==0]) # All negative reviews.
print(f'\n')
print(f'*** Positive video game reviews ***\n')
print(df[df.Sentiment_Classification_Labels==1]) # All positive reviews.
print(f'\n')

### Apply Named Entity Recognition (NER) to dataset ###
print(f'*** All named entities in video games dataset ***\n')
for i in range(len(video_review_texts)):  # Iterate through df rows.
    input_text = video_review_texts[i]
    chunks = ne_chunk(pos_tag(word_tokenize(input_text)))
    # Perform tokenisation, pos tagging and NER.
    for ne in chunks:
        if type(ne) == Tree: # Check current chunk is a named entity.
            print(ne)
print()