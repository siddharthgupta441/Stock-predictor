# import joblib
# import streamlit as st
# import joblib
# import numpy as np
# from data_training import get_embedding

# joblib.dump('stock_prediction_agent/model/clf.pkl')

# model = joblib.load("model/clf.pkl")
import sys
import time

def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor

spinner = spinning_cursor()
for _ in range(50):
    sys.stdout.write(next(spinner))
    sys.stdout.flush()
    time.sleep(0.1)
    sys.stdout.write('\b')