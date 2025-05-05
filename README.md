# Fake vs Real News Classification

This project focuses on building a machine learning model to classify news articles as **Fake** or **Real** using the **Gradient Boosting** algorithm.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation and Setup](#installation-and-setup)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training](#model-training)

## Project Overview

This project uses the **Gradient Boosting** model to classify news articles based on their title and text. The model is trained on the **Fake and Real News Dataset**. This dataset consists of news articles labeled as either **Fake** or **Real**. The goal is to train a model to accurately predict the label for unseen news articles.

## Dataset

The dataset consists of two CSV files:
- **Fake.csv**: Contains fake news articles.
- **True.csv**: Contains real news articles.

Each article contains the following columns:
- `title`: The title of the article.
- `text`: The body text of the article.
- `subject`: The subject/category of the article.
- `date`: The publication date.

We combine the `title` and `text` columns into a new feature, `content`, used for model training.

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/fake-vs-real-news-classification.git
   cd fake-vs-real-news-classification
