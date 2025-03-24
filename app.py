import streamlit as st
import nltk
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load PubMedBERT model for healthcare responses
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# NLP Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert text to lowercase and tokenize
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]  
    return " ".join(filtered_tokens)  # Return cleaned text

# Healthcare Chatbot Logic
def healthcare_chatbot(user_input):
    processed_input = preprocess_text(user_input)

    # Rule-based responses for common queries
    if "fever" in processed_input:
        return "For fever, stay hydrated, get rest, and consider paracetamol. If it persists, see a doctor."
    elif "headache" in processed_input:
        return "For headaches, drink water, rest, and avoid screen time. If severe, seek medical advice."
    elif "symptom" in processed_input:
        return "Please consult a doctor for an accurate diagnosis."
    elif "appointment" in processed_input:
        return "Would you like to schedule an appointment with a doctor?"
    elif "medicine" in processed_input:
        return "It is important to take prescribed medicines. Consult your doctor before taking any medication."
    
    # AI-generated response using PubMedBERT
    inputs = tokenizer(processed_input, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    return "I'm still learning. Please consult a doctor for medical advice."

# Streamlit Web App Interface
def main():
    st.title("Healthcare Assistant Chatbot 🤖💊")
    user_input = st.text_input("How can I assist you today?")

    if st.button("Submit"):
        if user_input:
            st.write("User:", user_input)

            with st.spinner("Processing your query... Please wait..."):
                response = healthcare_chatbot(user_input)

            st.write("Healthcare Assistant:", response)
        else:
            st.write("Please enter a message to get a response.")

# Run the app
if __name__ == "__main__":
    main()
