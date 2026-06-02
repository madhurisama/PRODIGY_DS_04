# PRODIGY_DS_04 — Sentiment Analysis

## Internship: Prodigy InfoTech — Data Science

## Task 04: Sentiment Analysis on Social Media Data

### Objective
Analyze and visualize sentiment patterns in 
social media data to understand public opinion 
and attitudes towards specific topics and brands.

### Dataset
Twitter Entity Sentiment Analysis — Kaggle
- twitter_training.csv
- twitter_validation.csv

### Steps Performed
- Loaded and merged training and validation data
- Cleaned tweet text:
  - Removed URLs, mentions, hashtags
  - Removed special characters and numbers
  - Converted to lowercase
- Extracted word count feature
- Built word frequency analysis
- Applied TF-IDF Vectorization (bigrams)
- Trained and compared 2 ML models
- Generated Word Clouds for each sentiment
- Built 12-plot visualization dashboard

### Models Used
- Multinomial Naive Bayes
- Linear Support Vector Classifier (SVC)

### Key Findings
- Linear SVC outperforms Naive Bayes
- Most tweets showed Positive sentiment
- Identified top positive and negative words
- Analyzed sentiment patterns across brands
- Word clouds revealed key sentiment drivers

### Visualizations
- Overall sentiment distribution (Pie + Bar)
- Sentiment % by brand (Stacked bar)
- Word Clouds (Positive, Negative, Neutral)
- Top 15 Positive and Negative words
- Tweet length by sentiment (Boxplot)
- Top 10 brands by tweet volume
- Model accuracy comparison
- Confusion matrix

### Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
- wordcloud

### Output
- sentiment_analysis.png — 12-plot dashboard

