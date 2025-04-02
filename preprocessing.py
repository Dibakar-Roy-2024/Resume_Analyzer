import pandas as pd 
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import  TfidfVectorizer
nltk.download('stopwords')
# load dataset
df = pd.read_csv('resume.csv')

# Text cleaning Funciton 
def clean_text(text):
    text= re.sub(r'\d+','', text)
    text = re.sub(r'[^\w\s]','',text)
    text = text.lower()
    text= ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text
df['cleaned_resume'] = df['Resume'].apply(clean_text)

#  Convert text tto numerrical forms 

vectorizer = TfidfVectorizer(max_features=500) 
X = vectorizer.fit_transform(df['cleaned_resume']).toarray()

# Save the preprocess data 
pd.DataFrame(X).to_csv("preprocessed_resume.csv", index = False)

print("✅ Preprocessing Done! Data saved as 'processed_resume.csv'.")