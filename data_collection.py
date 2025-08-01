import yfinance as yf
import pandas as pd
from newsapi import NewsApiClient
from datetime import date as d, timedelta
from dotenv import load_dotenv
import os

def news_data(query):
    api = os.getenv('api')
    from_date = d.today() - timedelta(days=30)
    to_date = d.today()

    news_list = []
    for date in pd.date_range(from_date, to_date):
        articles = api.get_everything(q=query, from_param=date.strftime("%Y-%m-%d"), to=date.strftime("%Y-%m-%d"), language='en', sort_by='relevancy')
        for a in articles['articles']:
            news_list.append([date.date(), a['title']])

    news_df = pd.DataFrame(news_list, columns=["Date", "Headline"])
    print("news_data function called with query:", query)
    return news_df


def yfinance_stock_price(ticker):
    try:
        #stock = yf.download(ticker, start="2024-01-01", end="2025-10-01")
        stock = yf.Ticker(ticker)
        stock = stock.history(period="1mo")
        #return stock_info.iloc[-1] if not stock_info.empty else None
        print("yfinance_stock_price function called with ticker:", ticker)
        return stock[['Close']]
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e} from Yahoo Finance")
        return None


print("Data collection module loaded successfully.")
# news_data = news_data('microsoft stock')
# print("News Data Shape:", type(news_data))

# ticker = 'MSFT'
# stock_df = yfinance_stock_price(ticker)
# print("Stock Data Shape:", type(stock_df))
# if stock_df is not None:
#     stock_df.reset_index(inplace=True)
#     stock_df['Date'] = pd.to_datetime(stock_df["Date"]).dt.date
#     print("Stock Data Date Range:", stock_df['Date'].min(), "to", stock_df['Date'].max())