from data_collection import news_data, yfinance_stock_price
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from yaspin import yaspin

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")

#stock_name = input("Enter the stock name: ")
stock_name = 'microsoft stock'
with yaspin(text="Loading news data...", color="cyan") as spinner:
    news_data = news_data(stock_name)
    if not news_data.empty:
        spinner.ok("✅")

#stock_code = input("Enter the stock code: ")
stock_code = 'MSFT'
with yaspin(text="Loading yfinance data...", color="cyan") as spinner:
    stock_df = yfinance_stock_price(stock_code)
    if not stock_df.empty:
        spinner.ok("✅")

if stock_df is not None:
    stock_df.reset_index(inplace=True)
    stock_df['Date'] = pd.to_datetime(stock_df["Date"]).dt.date

def combined_data_fun():
    combined_data = pd.merge(stock_df, news_data[['Date', 'Headline']], on='Date')
    combined_data.sort_values(by='Date', inplace=True)
    combined_data['Next_Close'] = combined_data['Close'].shift(-1)
    combined_data['Movement'] = (combined_data['Next_Close'] > combined_data['Close']).astype(int)
    combined_data.dropna(inplace=True)
    combined_data[['Date', 'Headline', 'Close', 'Next_Close', 'Movement']].head()
    return combined_data

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

#combined_data['embedding'] = combined_data['Headline'].apply(get_embedding)