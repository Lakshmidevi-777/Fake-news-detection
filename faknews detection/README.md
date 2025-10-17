# Fake News Detection - Streamlit App

This project detects whether a news article is **Real** or **Fake** using **Machine Learning + NLP**.

## Features
- Interactive web interface using Streamlit
- Uses TF-IDF vectorization and Logistic Regression
- Users can input news text and get instant predictions

## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the Streamlit app:
```bash
streamlit run app.py
```
3. Open the browser link provided and start testing your news.

## Dataset
- Use a CSV file named `news.csv` with `text` and `label` columns
- `label`: 0 = Real, 1 = Fake