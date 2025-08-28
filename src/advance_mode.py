# day3_fast_demo.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# 1️⃣ Load data
# -----------------------
df = pd.read_csv("data/heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# 2️⃣ Preprocessing
# -----------------------
numeric_features = ['age','trestbps','chol','thalach','oldpeak']
binary_features = ['sex','fbs','exang']
categorical_features = ['cp','restecg','slope','ca','thal']

numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
binary_transformer  = Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))])
categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                   ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('bin', binary_transformer, binary_features),
    ('cat', categorical_transformer, categorical_features)
])

# -----------------------
# 3️⃣ Gradient Boosting
# -----------------------
gbm = GradientBoostingClassifier(
    n_estimators=50,   # small, fast
    max_depth=3,
    random_state=42
)

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', gbm)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred))

# Save pipeline
with open("gbm_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# -----------------------
# 4️⃣ Tiny Transfer Learning Demo (optional)
# -----------------------
try:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input

    # Dummy image input for demonstration (32x32 RGB)
    dummy_input = np.random.rand(1,32,32,3)

    base_model = MobileNetV2(weights=None, include_top=False, input_tensor=Input(shape=(32,32,3)))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Quick forward pass to demonstrate
    pred = model.predict(dummy_input)
    print("Transfer learning dummy prediction:", pred[0][0])

except ImportError:
    print("TensorFlow not installed, skipping transfer learning demo.")
