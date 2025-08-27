import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

def load_data(path="data/heart.csv"):
    df = pd.read_csv(path)
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_models(X_train_scaled, X_train, y_train):
    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    return log_reg, dt, rf

def evaluate_model(y_test, y_pred, model_name):
    print(f"----- {model_name} -----")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("\n")

def plot_roc(y_test, model, X_test_scaled):
    y_probs = model.predict_proba(X_test_scaled)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model.__class__.__name__} (AUC = {roc_auc:.2f})')
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    log_reg, dt, rf = train_models(X_train_scaled, X_train, y_train)

    # Evaluate models
    evaluate_model(y_test, log_reg.predict(X_test_scaled), "Logistic Regression")
    evaluate_model(y_test, dt.predict(X_test), "Decision Tree")
    evaluate_model(y_test, rf.predict(X_test), "Random Forest")

    # ROC Curve for Logistic Regression
    plot_roc(y_test, log_reg, X_test_scaled)

    # Compare Accuracy
    results = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [
            accuracy_score(y_test, log_reg.predict(X_test_scaled)),
            accuracy_score(y_test, dt.predict(X_test)),
            accuracy_score(y_test, rf.predict(X_test))
        ]
    })
    print(results)
