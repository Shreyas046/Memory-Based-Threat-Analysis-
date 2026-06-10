import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def load_data(path):
    df = pd.read_csv(path)

    # ---- LABEL ----
    y = df['Class']

    # ---- DROP NON-FEATURES ----
    X = df.drop(columns=['Class', 'Category', 'Filename'])

    # ---- KEEP NUMERIC ----
    X = X.select_dtypes(include=['number'])

    # ---- ENCODE LABEL ----
    le = LabelEncoder()
    y = le.fit_transform(y)

    # ---- SCALE ----
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ---- SAVE FOR STREAMLIT ----
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le, "label_encoder.pkl")

    # ---- SPLIT ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test