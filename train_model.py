import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load your vector points dataset (CSV file with features and labels)
data = pd.read_csv('hand_landmarks.csv') 

# Separate features (X) and labels (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]   

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optionally, normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Train the Random Forest Classifier
# Initialize Random Forest with 100 trees (you can tweak this parameter)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Classifier Accuracy: {accuracy:.2f}')

# Optional: Get a detailed classification report
print(classification_report(y_test, y_pred))




# Save the trained model to a file
joblib.dump(rf_classifier, 'random_forest_model.pkl')

# Save the scaler as well (if used)
joblib.dump(scaler, 'scaler.pkl')