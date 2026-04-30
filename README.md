# Game Reviews Sentiment Analysis

Project adaptation from the **Natural Language Processing** (IOC6) module of the **Skills Bootcamp in Cloud Computing and Data Analytics** at Edge Hill University (2022).

## Project Overview
This project demonstrates Natural Language Processing (NLP) techniques to categorise video game reviews. It evaluates player sentiment and identifies key entities within the text.

## Technical Highlights
* **Sentiment Analysis:** Utilises `TextBlob` to calculate polarity scores for both individual sentences and full reviews.
* **Dynamic Classification:** Implements a custom thresholding logic to label reviews as "Positive" or "Negative" based on average sentiment polarities.
* **Named Entity Recognition (NER):** Uses `NLTK` tokenisation, POS tagging, and `ne_chunk` to extract and classify entities (organisations, locations, people) from raw text data.
* **Randomised Sampling:** Includes a helper function for sample iteration, enabling validation of NLP accuracy.

## Tech Stack
* **Python** (Core Logic)
* **Pandas** (Data Loading & DataFrame Management)
* **TextBlob** (NLP & Sentiment Evaluation)
* **NLTK** (Tokenisation, POS Tagging, & NER)
