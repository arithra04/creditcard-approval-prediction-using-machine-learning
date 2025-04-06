# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('credit_card_data.csv')

# Handling missing values
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  

# Drop Applicant_ID as it's not needed
df.drop(columns=['Applicant_ID'], inplace=True)

# Separate features and target variable
X = df.drop('Approval_Status', axis=1)
y = df['Approval_Status']

# Handle class imbalance with SMOTE
smote = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
rf_preds = rf_model.predict(X_test)

# Model Evaluation
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_preds))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# **Pause before asking user input**
input("\nPress Enter to continue and enter applicant details...")

# Function to take user input and predict approval status
def predict_credit_approval():
    print("\nEnter applicant details for credit approval prediction:")
    
    try:
        age = int(input("Age: "))
        income = float(input("Income: "))
        employment_status = input("Employment Status (Employed, Self-Employed, Unemployed, Retired): ").strip()
        credit_score = int(input("Credit Score: "))
        loan_amount = float(input("Loan Amount: "))
        existing_loans = int(input("Number of Existing Loans: "))
        debt_to_income_ratio = float(input("Debt-to-Income Ratio: "))

        # Encode employment status
        if employment_status in label_encoders['Employment_Status'].classes_:
            employment_status_encoded = label_encoders['Employment_Status'].transform([employment_status])[0]
        else:
            print("Invalid employment status! Using default value (Employed).")
            employment_status_encoded = label_encoders['Employment_Status'].transform(['Employed'])[0]

        # Create a DataFrame for new input
        new_applicant = pd.DataFrame([[age, income, employment_status_encoded, credit_score, loan_amount, existing_loans, debt_to_income_ratio]],
                                     columns=X.columns)

        # Scale input features
        new_applicant_scaled = scaler.transform(new_applicant)

        # Predict using trained model
        prediction = rf_model.predict(new_applicant_scaled)[0]


        # Display prediction result
        if prediction == 1:
            print("\n✅ Credit Approved! 🎉")
        else:
            print("\n❌ Credit Rejected.")

    except ValueError:
        print("\n⚠ Invalid input! Please enter numeric values where required.")

# Call the function to take user input
predict_credit_approval()
