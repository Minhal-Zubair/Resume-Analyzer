import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import make_pipeline
import pickle
import os

# Function to clean the dataset and handle duplicate columns
def clean_dataset(df):
    # Remove duplicate columns (keep the first occurrence)
    if df.columns.duplicated().any():
        st.warning("Duplicate columns found, renaming them...")
        df = df.loc[:, ~df.columns.duplicated()]  # Keep only the first occurrence of each column
    return df

# Function to load data and train the model
def load_data_and_train_model():
    # Load dataset (adjust the path if necessary)
    dataset_path = "clean_data1.csv"  # Update this to the actual path if needed
    if not os.path.exists(dataset_path):
        st.error(f"Dataset not found: {dataset_path}")
        return None, None, None, None

    df = pd.read_csv(dataset_path)
    
    # Clean the dataset to handle duplicates
    df = clean_dataset(df)

    # Show the first few rows of the cleaned dataset
    st.write("First few rows of the dataset:", df.head())

    # Example of a simple model setup with a TfidfVectorizer and Naive Bayes Classifier
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    model = MultinomialNB()

    # Make a pipeline to combine vectorizer and classifier
    pipeline = make_pipeline(vectorizer, model)

    # Assume the 'Resume' column contains text and 'Category' column is the label
    X = df['Resume']
    y = df['Category']

    # Train the model
    pipeline.fit(X, y)
    
    # Save the model to disk
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(pipeline, model_file)

    # Categories for prediction (you can expand this if needed)
    categories = df['Category'].unique()

    return pipeline, vectorizer, model, categories

# Function to load the trained model
def load_trained_model():
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    else:
        st.error("Model not found!")
        return None

# Function to predict the category of a resume text
def predict_category(model, resume_text):
    prediction = model.predict([resume_text])
    return prediction[0]

# Streamlit interface
def main():
    st.set_page_config(page_title="AI-Powered Resume Analyzer", layout="centered")
    
    # Header and subheader
    st.title("AI-Powered Resume Analyzer")
    st.subheader("Classify resumes into categories using machine learning")

    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Add some descriptive text
    st.markdown("""
        **How it works:**
        - Upload a CSV dataset with columns `Resume` and `Category` (this will be used for training).
        - Paste the resume text in the provided text box below.
        - The model will predict the category of the resume.
    """, unsafe_allow_html=True)
    
    # Load the model if available
    model = load_trained_model()

    if model is None:
        with st.spinner("Training model... Please wait. This may take a few moments..."):
            vectorizer, tfidf_transformer, classifier, categories = load_data_and_train_model()

            if vectorizer is not None:
                st.success("Model trained successfully!")
                model = load_trained_model()

    # User input section for testing the model
    st.markdown("<br>", unsafe_allow_html=True)
    resume_text = st.text_area("Paste the resume text here:", height=200)

    # Use a button to trigger prediction
    if st.button("Predict Category"):
        if resume_text:
            with st.spinner("Making prediction..."):
                category = predict_category(model, resume_text)
                st.write(f"**Predicted Category:** {category}")
        else:
            st.error("Please enter the resume text to predict the category.")

    
    
    # Footer (optional)
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    st.markdown("Made with ❤️ using Streamlit. Developed by Minhal.", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
