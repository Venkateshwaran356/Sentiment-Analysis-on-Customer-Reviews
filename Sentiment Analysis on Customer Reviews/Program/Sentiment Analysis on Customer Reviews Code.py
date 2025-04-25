





#---------------------------------------------------------------Open and Read Text File----------------------------------------------------------------------

with open("C:/Data Science and AI/Project/Sentiment Analysis on Customer Reviews/Datasets/Watches.txt", "r", encoding="utf-8") as file:
    text = file.read()

#----------------------------------------------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------Split into individual reviews----------------------------------------------------------------------

# Each review starts with 'product/productId:'
reviews = text.strip().split("product/productId:")[1:]  # skip the empty first element

# Add back the split token to each
reviews = ["product/productId:" + r.strip() for r in reviews]

#----------------------------------------------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------Extract key fields using regex----------------------------------------------------------------------

import re

data = []
for review in reviews:
    product_id = re.search(r"product/productId:\s*(.*)", review)
    title = re.search(r"product/title:\s*(.*)", review)
    user = re.search(r"review/userId:\s*(.*)", review)
    profile = re.search(r"review/profileName:\s*(.*)", review)
    score = re.search(r"review/score:\s*(.*)", review)
    summary = re.search(r"review/summary:\s*(.*)", review)
    text = re.search(r"review/text:\s*(.*)", review, re.DOTALL)

    data.append({
        "product_id": product_id.group(1) if product_id else "",
        "title": title.group(1) if title else "",
        "user_id": user.group(1) if user else "",
        "profile_name": profile.group(1) if profile else "",
        "score": float(score.group(1)) if score else None,
        "summary": summary.group(1) if summary else "",
        "review_text": text.group(1).strip() if text else ""
    })


#----------------------------------------------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------Create a DataFrame and Export----------------------------------------------------------------------

import pandas as pd

df = pd.DataFrame(data)
df.to_csv('Customer_Review_Dataset.csv', index=False)

#----------------------------------------------------------------------------------------------------------------------------------------------------





#----------------------------------------------------Import Libraries and Perform Sentiment Analysis----------------------------------------------------------------------

from textblob import TextBlob

# Define a function to calculate sentiment polarity
def get_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    # Return the polarity of the review text (-1 to 1 scale)
    return blob.sentiment.polarity

# Apply the sentiment function to the 'review_text' column
df['sentiment'] = df['review_text'].apply(get_sentiment)

#----------------------------------------------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------Interpret the Sentiment Score----------------------------------------------------------------------

def categorize_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Create a new column for sentiment category
df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)

#----------------------------------------------------------------------------------------------------------------------------------------------------





#--------------------------------------------------     ---Visualize the Sentiment Distribution----------------------------------------------------------------------

import matplotlib.pyplot as plt

# Plot sentiment distribution
sentiment_counts = df['sentiment_category'].value_counts()

plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution of Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------Tokenization + Stopword Removal + Lemmatization----------------------------------------------------------------------

# import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download necessary NLTK data
"""
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
"""

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Tokenize
    words = word_tokenize(text)
    
    # 4. Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # 5. Join back to a cleaned string
    return ' '.join(words)

# Apply to your DataFrame
df['cleaned_review'] = df['review_text'].apply(preprocess_text)

df.to_csv("customer_reviews_with_sentiment.csv", index=False)

#----------------------------------------------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------TF-IDF Vectorization----------------------------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Limit the number of features to 5000

X = df['cleaned_review']
Y = df['sentiment_category']

# Fit and transform the cleaned reviews
tfidf_X = vectorizer.fit_transform(X)

#----------------------------------------------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------Model Building----------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(tfidf_X, Y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
Y_train_pred = model.predict(X_train) 
Y_test_pred = model.predict(X_test) 

#----------------------------------------------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------Model Evaluation---------------------------------------------------------------------

# Evaluate Model Performance
train_accuracy = accuracy_score(y_train, Y_train_pred)
test_accuracy = accuracy_score(y_test, Y_test_pred)
print(f"Trained Logistic Regression Accuracy: {train_accuracy:.4f}")
print(f"Tested Logistic Regression Accuracy: {test_accuracy:.4f}")

#----------------------------------------------------------------------------------------------------------------------------------------------------






#---------------------------------------------------------------Save the Model---------------------------------------------------------------------

import joblib

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

#----------------------------------------------------------------------------------------------------------------------------------------------------




