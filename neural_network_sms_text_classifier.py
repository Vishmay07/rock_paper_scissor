# Cell 1: Import Libraries
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Cell 2: Load and Preprocess Data
# Load the dataset (SMS Spam Collection)
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms_spam.csv"  # Replace with actual path or URL
df = pd.read_csv(url, encoding='latin-1')
df = df[['v1', 'v2']]  # Selecting only the columns of interest
df.columns = ['label', 'message']  # Rename columns for easier access

# Explore the first few rows of the dataset
df.head()

# Convert labels 'ham' to 0 and 'spam' to 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Cell 3: Build the Model
# Use a TfidfVectorizer and Multinomial Naive Bayes in a pipeline
model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())

# Train the model on the training data
model.fit(X_train, y_train)

# Cell 4: Define the Prediction Function
def predict_message(message: str):
    """
    Predict if a message is spam or ham.

    Args:
    message (str): The SMS message to classify.

    Returns:
    List: The first element is the probability of being spam (0 or 1), and the second element is the classification ('ham' or 'spam').
    """
    # Make a prediction using the trained model
    pred_prob = model.predict_proba([message])[0][1]  # Probability of spam (class 1)
    label = 'spam' if pred_prob >= 0.5 else 'ham'
    
    # Return the probability and label as a list
    return [pred_prob, label]

# Cell 5: Model Evaluation
# Evaluate the model on the test set
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Example Test
# Let's check predictions for a few test messages
test_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Click here to claim it.",
    "Hey, are you coming to the party tonight?",
    "Your account has been compromised. Please click here to reset your password.",
    "Can you send me the details of the meeting tomorrow?",
]

for message in test_messages:
    print(f"Message: {message}")
    print(f"Prediction: {predict_message(message)}\n")
