import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
url = "https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD"
df = pd.read_csv(url)

# Select features
feature_cols = ['Electric Range', 'Model Year', 'Make', 'Model', 'County', 'City']
X = df[feature_cols].copy()
y = df['Electric Vehicle Type']

# Encode categorical features
categorical_features = ['Make', 'Model', 'County', 'City']
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Encode target
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features
scaler = StandardScaler()
X_train[['Electric Range', 'Model Year']] = scaler.fit_transform(X_train[['Electric Range', 'Model Year']])
X_test[['Electric Range', 'Model Year']] = scaler.transform(X_test[['Electric Range', 'Model Year']])

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)

# Export model
joblib.dump(rf_model, "random_forest_ev_model.pkl")
print("✅ Model exported successfully as random_forest_ev_model.pkl")
