import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

data = pd.read_csv("data/creditcard.csv")
X = data.drop("Class", axis=1)
y = data["Class"]

model = joblib.load("models/model.pkl")

y_pred = model.predict(X)

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Fraud Detection Confusion Matrix")
plt.show()
