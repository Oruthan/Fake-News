import pandas as pd
import string
import re
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

print("Starting model training...")

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

try:
    # Load the datasets
    print("Loading datasets...")
    data_fake = pd.read_csv('data/Fake.csv')
    data_true = pd.read_csv('data/True.csv')

    # Add class labels
    data_fake["class"] = 0  # Fake news class
    data_true["class"] = 1  # Real news class

    # Remove some rows for manual testing
    data_fake_manual_testing = data_fake.tail(10)
    data_true_manual_testing = data_true.tail(10)
    data_fake = data_fake.iloc[:-10]
    data_true = data_true.iloc[:-10]

    # Combine the datasets
    data_merge = pd.concat([data_fake, data_true], axis=0)
    data = data_merge.drop(['title', 'subject', 'date'], axis=1)  # Drop unnecessary columns

    # Shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Apply text preprocessing to the 'text' column
    print("Preprocessing text data...")
    data['text'] = data['text'].apply(wordopt)

    # Define features and target variable
    x = data['text']
    y = data['class']

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    # Vectorize the text data
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer()
    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)

    # Initialize and train the models
    print("Training Logistic Regression model...")
    LR = LogisticRegression()
    LR.fit(xv_train, y_train)

    print("Training Decision Tree model...")
    DT = DecisionTreeClassifier()
    DT.fit(xv_train, y_train)

    print("Training Gradient Boosting model...")
    GB = GradientBoostingClassifier(random_state=0)
    GB.fit(xv_train, y_train)

    print("Training Random Forest model...")
    RF = RandomForestClassifier(random_state=0)
    RF.fit(xv_train, y_train)

    # Make predictions
    pred_lr = LR.predict(xv_test)
    pred_dt = DT.predict(xv_test)
    pred_gb = GB.predict(xv_test)
    pred_rf = RF.predict(xv_test)

    # Calculate accuracy
    acc_lr = accuracy_score(y_test, pred_lr)
    acc_dt = accuracy_score(y_test, pred_dt)
    acc_gb = accuracy_score(y_test, pred_gb)
    acc_rf = accuracy_score(y_test, pred_rf)

    # Print accuracy results
    print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
    print(f"Decision Tree Accuracy: {acc_dt:.4f}")
    print(f"Gradient Boosting Accuracy: {acc_gb:.4f}")
    print(f"Random Forest Accuracy: {acc_rf:.4f}")

    # Save the models
    print("Saving models...")
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    with open('models/model_lr.pkl', 'wb') as f:
        pickle.dump(LR, f)

    with open('models/model_dt.pkl', 'wb') as f:
        pickle.dump(DT, f)

    with open('models/model_gb.pkl', 'wb') as f:
        pickle.dump(GB, f)

    with open('models/model_rf.pkl', 'wb') as f:
        pickle.dump(RF, f)

    print("Model training and saving complete!")

except Exception as e:
    print(f"Error during model training: {e}")


    



import pickle
import os

# Create a models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the TF-IDF vectorizer
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save each model
with open('models/model_lr.pkl', 'wb') as f:
    pickle.dump(LR, f)
    
with open('models/model_dt.pkl', 'wb') as f:
    pickle.dump(DT, f)
    
with open('models/model_gb.pkl', 'wb') as f:
    pickle.dump(GB, f)
    
with open('models/model_rf.pkl', 'wb') as f:
    pickle.dump(RF, f)

print("Models saved successfully!")