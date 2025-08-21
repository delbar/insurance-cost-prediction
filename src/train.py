# 📌 train.py - پیش‌بینی هزینه بیمه سلامت

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# 1. بارگذاری دیتاست
# -------------------------------
data_path = r"C:/Users/ZBook Fury/insurance-cost-prediction/data/insurance_salamat.csv"
df = pd.read_csv(data_path)

print("شکل دیتاست:", df.shape)
print(df.head())

# -------------------------------
# 2. پیش‌پردازش
# -------------------------------
df = pd.get_dummies(df, drop_first=True)

# -------------------------------
# 3. جدا کردن ویژگی‌ها و هدف
# -------------------------------
X = df.drop("charges", axis=1)
y = df["charges"]

# -------------------------------
# 4. تقسیم داده‌ها به Train و Test
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 5. مدل Linear Regression
# -------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nLinear Regression Results:")
print("R²:", r2_score(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))

# -------------------------------
# 6. مدل Random Forest
# -------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Results:")
print("R²:", r2_score(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))

# -------------------------------
# 7. اهمیت ویژگی‌ها
# -------------------------------
importances_series = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(10,6))
sns.barplot(x=importances_series.values, y=importances_series.index)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()

# اطمینان از وجود پوشه results
os.makedirs(r"C:/Users/ZBook Fury/insurance-cost-prediction/results", exist_ok=True)

# ذخیره نمودار
output_path = r"C:/Users/ZBook Fury/insurance-cost-prediction/results/feature_importance.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"✅ نمودار ذخیره شد در: {output_path}")
