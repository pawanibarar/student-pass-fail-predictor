# --------------------------------------------
# STUDENT PASS/FAIL PREDICTOR (WORKING VERSION)
# --------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Student Details
student_name = input("Enter name")
subject = "Artificial Intelligence"

print("---------------------------------")
print("STUDENT PERFORMANCE PREDICTOR")
print("---------------------------------")
print("Student Name:", student_name)
print("Subject:", subject)
print("---------------------------------\n")

print("--- Student Pass/Fail Predictor ---")

# Step 1: Load your CSV file
csv_path = "students_ai.csv"   # <-- Correct CSV name

try:
    df = pd.read_csv(csv_path)
    print("CSV loaded successfully!\n")
except:
    print("Error: Could not load CSV. Check the file name.")
    exit()

# Step 2: Display first rows
print("Preview of your data:")
print(df.head(), "\n")

# Step 3: Encode result column
df["Result"] = df["Result"].map({"Pass": 1, "Fail": 0})

# Step 4: Choose features
features = ["Study_Hours", "Attendance", "Internal_Marks"]
label = "Result"

X = df[features]
y = df[label]

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Test accuracy
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Predict for new student
print("\n--- Enter student details to predict Pass/Fail ---")

study = float(input("Study Hours per day: "))
attendance = float(input("Attendance %: "))
internal = float(input("Internal Marks: "))

# Create DataFrame with proper feature names
new_data = pd.DataFrame(
    [[study, attendance, internal]],
    columns=["Study_Hours", "Attendance", "Internal_Marks"]
)

prediction = model.predict(new_data)

# Step 9: Final result
if prediction[0] == 1:
    print("\nPredicted Result: PASS ✅")
else:
    print("\nPredicted Result: FAIL ❌")
