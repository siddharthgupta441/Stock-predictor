import joblib
from data_training import get_embedding
from yaspin import yaspin
from data_collection import news_data
import data_preprocessing as dp

joblib.dump('stock_prediction_agent/model/clf.pkl')

model = joblib.load("model/clf.pkl")
stock_name = dp.stock_name

with yaspin(text="Loading news data...", color="cyan") as spinner:
    news_data = news_data(stock_name)
    if not news_data.empty:
        spinner.ok("✅")

with yaspin(text="Training the data...", color='cyan') as spinner:
    embedding = get_embedding(news_data['Headline'])
    if not embedding.empty:
        spinner.ok("✅")

prediction = model.predict(embedding)[0]
proba = model.predict_proba(embedding)[0][prediction]

if prediction == 1:
    print(f"Prediction: Price Likely to GO UP (Confidence: {proba:.2f})")
else:
    print(f"Prediction: Price Likely to GO DOWN (Confidence: {proba:.2f})")