import data_preprocessing as dp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

print("Data training module loaded successfully.")
combined_data = dp.combined_data
x = np.stack(combined_data['embedding'].values)
y = combined_data['Movement'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))