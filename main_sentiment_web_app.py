import streamlit as st
import pandas as pd
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import speech_recognition as sr
import pyttsx3

# Download NLTK 
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Initialize NLTK
stemmer = PorterStemmer()
stopwords_list = set(stopwords.words('english'))

# Function to preprocess text
def Text_preprocessor(review):
    # Remove HTML tags
    review = re.sub(r'<.*?>', '', review)
    # Remove punctuation
    review = review.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    review = review.translate(str.maketrans('', '', string.digits))
    # Lowercase all letters
    review = review.lower()
    # Tokenize the text
    words = nltk.word_tokenize(review)
    # Remove stop words and perform stemming
    review = ' '.join([stemmer.stem(word) for word in words if word.lower() not in stopwords_list])

    return review

# Load trained model and vectorizer which we create in colab environment
transform_path = "C:\\Predict_Sentiment\\transform.pkl"
model_path = "C:\\Predict_Sentiment\\sentiment-prediction-model.pkl"

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)
with open(transform_path, "rb") as transform_file:
    tfidf_vectorizer = pickle.load(transform_file)

# labels for sentiment classes
labels = ['Negative', 'Neutral', 'Positive']

# Function to get sentiment prediction
def get_sentiment(review):
    x = Text_preprocessor(review)
    X = tfidf_vectorizer.transform([x])
    y = int(model.predict(X.reshape(1, -1)))
    return labels[y]

# Initialize the recognizer
r = sr.Recognizer()

# Function to convert text to speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command) 
    engine.runAndWait()

# Streamlit app
def main():
    st.set_page_config(
        page_title="Sentiment Prediction Web App",
        page_icon=":🙂:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Sentiment Analysis App :heart_eyes:")

    # Radio buttons for input selection
    input_option = st.sidebar.radio("Choose Input Option:", ("Text Input", "CSV File Input", "Speech Input"))

    if input_option == "Text Input":
        st.header("Text Input :memo:")
        review = st.text_area("Enter your review:")

        if st.button("Analyze Text"):
            if review:
                sentiment = get_sentiment(review)
                st.success(f"The sentiment is: {sentiment}")
            else:
                st.warning("Please enter a review.")

    elif input_option == "CSV File Input":
        st.header("CSV File Input :file_folder:")
        csv_file = st.file_uploader("Upload CSV file", type=['csv'], key='csv_uploader')

        if csv_file is not None:
            df = pd.read_csv(csv_file)

            # Show/Hide preview of CSV file
            show_preview = st.checkbox("Show Preview")
            if show_preview:
                st.write("Preview of the CSV file:")
                st.dataframe(df.head())
            
            # Perform sentiment analysis on the 'review' column
            df['sentiment'] = df['review'].apply(get_sentiment)

            # Download CSV button
            st.sidebar.markdown("---")
            st.sidebar.markdown(
                f"Download Updated CSV :arrow_down:",
                unsafe_allow_html=True
            )
            st.sidebar.download_button(
                label="Download",
                data=df.to_csv().encode('utf-8'),
                file_name='updated_sentiment_analysis.csv',
                mime='text/csv'
            )

            # Display the DataFrame with sentiment analysis results
            st.write("Sentiment analysis results:")
            st.dataframe(df.style.set_properties(subset=['review'], **{'width': '800px'}))

    elif input_option == "Speech Input":
        st.header("Speech Input :microphone:")
        if st.button("Start Speaking"):
            try:
                with sr.Microphone() as source:
                    st.info("Listening... Speak something.")

                    # Adjust for ambient noise
                    r.adjust_for_ambient_noise(source, duration=0.2)

                    # Listen for audio input
                    audio = r.listen(source)

                    # Recognize audio using Google
                    text = r.recognize_google(audio)
                    text = text.lower()

                    # Display recognized text
                    st.success(f"You said: {text}")

                    # Perform sentiment analysis
                    sentiment = get_sentiment(text)
                    st.success(f"The sentiment is: {sentiment}")

            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")
                
            except sr.UnknownValueError:
                st.warning("Unknown error occurred")

if __name__ == "__main__":
    main()
