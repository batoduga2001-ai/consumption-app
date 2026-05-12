# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# =========================
# 2. LOAD DATASET
# =========================
df = pd.read_csv("telco.csv")

# =========================
# 3. CLEAN DATA
# =========================

# remove ID column
df.drop("customerID", axis=1, inplace=True)

# fix TotalCharges (convert string → numeric)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# =========================
# 4. ENCODE ALL CATEGORICAL DATA (IMPORTANT FIX)
# =========================

df = pd.get_dummies(df, drop_first=True)

# =========================
# 5. SPLIT FEATURES AND TARGET
# =========================

# After get_dummies, target becomes Churn_Yes
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# =========================
# 6. TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 7. TRAIN MODEL
# =========================

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# 8. SAVE MODEL (.PKL FILE)
# =========================

joblib.dump(model, "telco_churn_model.pkl")

print("✅ Model trained successfully!")
print("✅ Saved as telco_churn_model.pkl")