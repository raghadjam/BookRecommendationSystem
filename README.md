# Book Recommender System - Neural Collaborative Filtering (NCF)

## Overview  
This project implements a Book Recommender System using Neural Collaborative Filtering (NCF). The model combines Matrix Factorization (MF) and Multilayer Perceptron (MLP) to learn user-book interactions.  

## Dataset  
[Kaggle - Book Recommendation Dataset](https://www.kaggle.com/datasets/ra4u12/bookrecommendation?select=BX-Book-Ratings.csv)  
- BX-Books.csv → Book metadata (ISBN, Title, Author).  
- BX-Book-Ratings.csv → User ratings (0–10).  
- BX-Users.csv → User demographics.  

## Preprocessing:  
- Filtered Users (<150 ratings) & Books (<200 ratings).  
- Mapped ISBN & User-ID to integers.  
- Balanced Data: Removed zero interactions.  

## Model  
Built using TensorFlow/Keras with:  
- MF Layer: Embeddings & dot product.  
- MLP Layer: Concatenated embeddings → Dense(8, ReLU) → Dense(4, ReLU).  
- Prediction: MF & MLP outputs → Sigmoid activation.  

## Performance  
 

| Accuracy | Precision | Recall | F1 Score |  
|----------|-----------|--------|----------|  
|   0.87   |   0.90    |  0.96  |   0.93   |  


