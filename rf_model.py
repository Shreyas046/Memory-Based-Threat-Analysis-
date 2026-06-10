from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_rf(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Random Forest Accuracy: {acc * 100:.2f}%")

    return model