import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0

def voice_input():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.write("Please speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        st.write("Recognizing...")
        input_text = recognizer.recognize_google(audio)
        st.write(f"You said: {input_text}")
        return input_text
    except sr.UnknownValueError:
        st.write("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError:
        st.write("Could not request results from Google Speech Recognition service.")
        return None

def main():
    global counter
    st.title("Intents of Chatbot using NLP")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message or use voice input to start the conversation.")
        
        # Voice input option
        voice_input_button = st.button("Speak")
        if voice_input_button:
            user_input = voice_input()
        else:
            counter += 1
            user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            # Convert the user input to a string
            user_input_str = str(user_input)
            response = chatbot(user_input)
            
            # Display user message in chat bubble style
            st.markdown(f'<div style="background-color: #007BFF; color: white; padding: 10px; border-radius: 20px; max-width: 80%; margin: 5px; word-wrap: break-word; align-self: flex-start;">'
                        f'<b>You:</b> {user_input_str}</div>', unsafe_allow_html=True)

            # Display chatbot message in bubble style
            st.markdown(f'<div style="background-color: #00C851; color: white; padding: 10px; border-radius: 20px; max-width: 80%; margin: 5px; word-wrap: break-word; align-self: flex-end;">'
                        f'<b>Chatbot:</b> {response}</div>', unsafe_allow_html=True)

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Check if the chat_log.csv file exists, and if not, create it with column names
            if not os.path.exists('chat_log.csv'):
                with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")

        # Check if the chat_log.csv file exists and has content
        if os.path.exists('chat_log.csv') and os.path.getsize('chat_log.csv') > 0:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                history = list(csv_reader)
                history.reverse()  # Reverse to display the most recent first

                if history:
                    # Skip header row if present and then iterate through the history
                    for row in history[1:]:  # Skip the header row in the displayed history
                        st.text(f"User: {row[0]}")
                        st.text(f"Chatbot: {row[1]}")
                        st.text(f"Timestamp: {row[2]}")
                        st.markdown("---")
                else:
                    st.write("No conversation history available.")
        else:
            st.write("No conversation history available.")

    # About Menu
    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.")

        st.subheader("Project Overview:")
        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface. The interface allows users to input text and receive responses from the chatbot.
        """)

        st.subheader("Dataset:")
        st.write("""
        The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
        - Intents: The intent of the user input (e.g. "greeting", "budget", "about")
        - Entities: The entities extracted from user input (e.g. "Hi", "How do I create a budget?", "What is your purpose?")
        - Text: The user input text.
        """)

        st.subheader("Streamlit Chatbot Interface:")
        st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

        st.subheader("Conclusion:")
        st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms.")

if __name__ == '__main__':
    main()
