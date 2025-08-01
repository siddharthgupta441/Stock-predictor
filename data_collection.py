import yfinance as yf
import pandas as pd
from newsapi import NewsApiClient
from datetime import date as d, timedelta
from dotenv import load_dotenv
import os

def news_data(query):
    load_dotenv()
    api = os.getenv('api')
    api = NewsApiClient(api_key=api)
    from_date = d.today() - timedelta(days=30)
    to_date = d.today()

    news_list = []
    for date in pd.date_range(from_date, to_date):
        articles = api.get_everything(q=query, from_param=date.strftime("%Y-%m-%d"), to=date.strftime("%Y-%m-%d"), language='en', sort_by='relevancy')
        for a in articles['articles']:
            news_list.append([date.date(), a['title']])

    news_df = pd.DataFrame(news_list, columns=["Date", "Headline"])
    return news_df


def yfinance_stock_price(ticker):
    try:
        #stock = yf.download(ticker, start="2024-01-01", end="2025-10-01")
        stock = yf.Ticker(ticker)
        stock = stock.history(period="1mo")
        #return stock_info.iloc[-1] if not stock_info.empty else None
        return stock[['Close']]
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e} from Yahoo Finance")
        return None