import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

def train_and_save_model():
    # 1️⃣ Load Dataset
    df = pd.read_csv("data/heart.csv")

    # 2️⃣ Split features & target
    X = df.drop('target', axis=1)
    y = df['target'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3️⃣ Preprocessing
    numeric_features = ['age','trestbps','chol','thalach','oldpeak']
    binary_features  = ['sex','fbs','exang']
    categorical_features = ['cp','restecg','slope','ca','thal']

    numeric_transformer = Pipeline([
        ('imputer',SimpleImputer(strategy='median')), 
        ('scaler',StandardScaler())
    ])
    binary_transformer  = Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent'))
    ])
    categorical_transformer = Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('bin', binary_transformer, binary_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # 4️⃣ Model + GridSearch
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'clf__n_estimators': [100,200],
        'clf__max_depth': [3,5,7],
        'clf__min_samples_split':[2,5,10]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5)
    grid.fit(X_train, y_train)

    # 5️⃣ Evaluation
    y_pred = grid.predict(X_test)
    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, grid.predict_proba(X_test)[:,1]))

    # 6️⃣ Save Model
    with open("heart_pipeline.pkl","wb") as f:
        pickle.dump(grid.best_estimator_, f)

if __name__ == "__main__":
    train_and_save_model()
