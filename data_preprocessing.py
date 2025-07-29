from data_collection import news_data, yfinance_stock_price
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
import torch

stopword = set(stopwords.words('english'))

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")

news_data = news_data('microsoft stock')
#news_data = preprocess_news_data(news_data)

stock_df = yfinance_stock_price('MSFT')
if stock_df is not None:
    #print("Stock Data Shape:", type(stock_df))
    stock_df.reset_index(inplace=True)
    stock_df['Date'] = pd.to_datetime(stock_df["Date"]).dt.date

#print(f"new data: {type(news_data)}, stock data: {type(stock_df)}")
combined_data = pd.merge(stock_df, news_data[['Date', 'Headline']], on='Date')
combined_data.sort_values(by='Date', inplace=True)
combined_data['Next_Close'] = combined_data['Close'].shift(-1)
combined_data['Movement'] = (combined_data['Next_Close'] > combined_data['Close']).astype(int)
combined_data.dropna(inplace=True)
combined_data[['Date', 'Headline', 'Close', 'Next_Close', 'Movement']].head()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

combined_data['embedding'] = combined_data['Headline'].apply(get_embedding)
#print(combined_data[['Date', 'Headline', 'embedding', 'Movement']].head())
print("Data preprocessing module loaded successfully.")
