# Sentiment-Analysis-with-BERT

This project explores sentiment analysis using a pre-trained BERT model (nlptown/bert-base-multilingual-uncased-sentiment) to classify textual data into sentiment categories.

## Project Overview
The project implements a robust pipeline for analyzing textual reviews:

- **Model:** Pre-trained BERT from Hugging Face.
- **Objective:** Classify sentiment in customer reviews.
- **Tools:** Python, Transformers, BeautifulSoup, and Pandas.

# Key Steps

**Model Initialization:**
```python
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
```

**Tokenization**

```python
tokens = tokenizer.encode("What a worst of resources", return_tensors='pt')
```

**Data Collection:** Web scraping and preprocessing textual reviews using BeautifulSoup.

**Sentiment Calculation:** Utilizing model outputs for sentiment predictions.
