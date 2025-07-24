# Hotel Review Sentiment Analysis

This project presents a comparative analysis of sentiment classification techniques applied to hotel customer reviews. The goal is to classify reviews as **positive** or **negative** using three different models: Logistic Regression, Random Forest, and Bidirectional LSTM (BiLSTM). I wanted to learn and understand how accuracy differes from model to model and how SMOTE (Synthetic Minority Oversampling Technique) can be used in imbalanced datasets. 


---

## Objective

To implement and compare the performance of multiple machine learning models for sentiment classification, using a real-world dataset of hotel reviews. The analysis emphasizes both traditional and neural network-based approaches to understand model effectiveness in natural language processing (NLP) tasks.

---

## Models Implemented

| Model              | Category           | Description                                |
|--------------------|--------------------|--------------------------------------------|
| Logistic Regression | Supervised Learning | Baseline classifier with linear decision boundary |
| Random Forest       | Ensemble Learning   | Tree-based model robust to overfitting     |
| Bidirectional LSTM  | Deep Learning       | Captures contextual relationships in text sequences |

---

## Evaluation Metrics

Each model was assessed using the following performance metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC AUC Score**
- **Confusion Matrix**

All models were trained and tested on the same dataset split to ensure a fair and consistent comparison.

---

## Results Summary

| Metric      | Logistic Regression | Random Forest | Bidirectional LSTM |
|-------------|---------------------|---------------|---------------------|
| Accuracy    | Moderate             | Good          | Excellent           |
| F1-Score    | Fair                 | Strong        | Very Strong         |
| ROC-AUC     | Acceptable           | High          | Very High           |

The BiLSTM model achieved the highest performance across all evaluation metrics, indicating its strength in capturing the semantic and sequential patterns within textual data.

---

## Model Saving

- Logistic Regression and Random Forest models were serialized using `joblib`
- The BiLSTM model was saved as a `.h5` file using Keras
- All saved models are production-ready and can be integrated into web applications or ML pipelines

---

## Technologies Used

- Python (Pandas, NumPy, Scikit-learn)
- Natural Language Toolkit (NLTK)
- TensorFlow / Keras
- Matplotlib & Seaborn (for visualization)
- Jupyter Notebook


