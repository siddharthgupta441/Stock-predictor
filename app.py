# import joblib
# import streamlit as st
# import joblib
# import numpy as np
# from data_training import get_embedding

# joblib.dump('stock_prediction_agent/model/clf.pkl')

# model = joblib.load("model/clf.pkl")
from yaspin import yaspin
import time

with yaspin(text="Loading...", color="cyan") as spinner:
    time.sleep(5)  # simulate loading
    spinner.ok("✅ ") 