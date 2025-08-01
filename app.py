import joblib
import streamlit as st
import joblib
import numpy as np
from data_training import get_embedding

joblib.dump('stock_prediction_agent/model/clf.pkl')

model = joblib.load("model/clf.pkl")
 #in developing phase