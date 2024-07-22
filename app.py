import os
import streamlit as st
from pathlib import Path
import PyPDF2
import pathway as pw
import google.generativeai as genai
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter
import pandas as pd

# Configure the API key for Gemini
genai.configure(api_key=os.getenv('YOUR-GEMINI-API-KEY'))

# Initialize the model
model = genai.GenerativeModel('gemini-1.0-pro-latest')

# Define the custom PDF connector
class PDFConnector:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def read(self):
        try:
            with open(self.file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                content = ""
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    content += page.extract_text()
            return content
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""

# Define a UDF to extract relevant keywords
@pw.udf
def extract_relevant_keywords(text: str, question: str) -> str:
    words = text.split()
    question_words = set(question.lower().split())
    filtered_words = [word for word in words if word.lower() not in ENGLISH_STOP_WORDS and word.lower() in question_words]
    most_common_words = Counter(filtered_words).most_common(10)
    keywords = ', '.join([word for word, _ in most_common_words])
    return keywords

def get_gemini_answer(context, question):
    try:
        prompt = f"Context: {context}\nQuestion: {question}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error getting answer from Gemini: {e}")
        return "Error: Could not get answer from Gemini"

def answer_questions(pdf_content, questions):
    answers = []
    for question in questions:
        answer = get_gemini_answer(pdf_content, question)
        answers.append(answer)
    return answers

def main():
    st.title("PDF Question Answering App with Pathway and Gemini")
    
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Save the uploaded file
        file_path = Path(f"./uploads/{uploaded_file.name}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Use the custom PDF connector to read PDF content
        pdf_connector = PDFConnector(file_path)
        pdf_content = pdf_connector.read()
        st.text_area("PDF Content", pdf_content, height=300)
        
        questions = st.text_area("Enter your questions (one per line)").split("\n")
        
        if st.button("Get Answers"):
            if pdf_content and questions:
                # Process PDF content with Pathway to extract relevant keywords
                pdf_data = pw.Table.from_pandas(pd.DataFrame({'content': [pdf_content], 'question': [questions[0]]}))
                processed_data = pdf_data.with_columns(keywords=extract_relevant_keywords(pdf_data.content, pdf_data.question))
                processed_content = processed_data.to_pandas().iloc[0, 2]
                st.text_area("Extracted Keywords", processed_content, height=100)
                
                answers = answer_questions(pdf_content, questions)
                for q, a in zip(questions, answers):
                    st.write(f"**Q:** {q}\n**A:** {a}\n")

if __name__ == "__main__":
    main()

