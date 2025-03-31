
# Flipkart Product Review Sentiment Analysis



## Project Overview
This project focuses on sentiment analysis of Flipkart product reviews. Using Natural Language Processing (NLP) techniques and machine learning, the model classifies reviews as positive or negative based on their ratings and textual content. The analysis involves data preprocessing, visualization, and model training using a Decision Tree classifier.
## Dataset
Source: Flipkart product reviews dataset (Kaggle)
## Features and Techniques Used
 Data Cleaning & Preprocessing

    1. Tokenization using NLTK

    2. Stopword removal

    3. Lowercasing and punctuation removal

Data Visualization

    1. Distribution of ratings

    2. WordClouds for positive and negative reviews

    3. Feature importance visualization

Feature Extraction

    1. TF-IDF vectorization (max 2500 features)

Machine Learning Model

    1. Decision Tree Classifier

    2. Training-Test split (67%-33%)

    3. Model evaluation using accuracy score and confusion matrix
## Installation & Dependencies
install the required Python libraries:

    pip install pandas nltk matplotlib seaborn wordcloud scikit-learn
Additionally, download NLTK resources:

    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')    
    nltk.download('punkt_tab')
## Preprocessing & Execution

    1. Load the dataset: Ensure flipkart_data.csv is in the working directory.

    2. Preprocess the data: Tokenization, stopword removal, and text cleaning.

    3. Visualize insights: Generate rating distribution, sentiment count, and word clouds.

    4. Train the model: Decision Tree classification on TF-IDF transformed text data.

    5. Evaluate performance: Check accuracy and confusion matrix.
## Visualizations

    Rating Distribution: Displays how ratings are distributed among reviews.

    Sentiment Count Plot: Shows the proportion of positive vs. negative reviews.

    WordClouds: Highlight frequently used words in positive and negative reviews.

    Feature Importance: Identifies key words influencing the sentiment classification.
## Results
    Model Accuracy: Displays training accuracy of the Decision Tree classifier.

    Confusion Matrix: Visual representation of classification performance.
## Future Improvements
    Experiment with advanced models (e.g., Random Forest, SVM, Neural Networks)

    Hyperparameter tuning for improved accuracy

    Implement deep learning with word embeddings

    Expand dataset for better generalization