import streamlit as st
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from PyPDF2 import PdfReader
import os
import torch
from transformers import pipeline

st.set_page_config(layout="wide")

# Specify the local path to your model directory
model_dir = r"C:\Users\Dell\Desktop\TRAINED_MODLES"

# Check if the model directory exists
if not os.path.exists(model_dir):
    st.error(f"The model directory {model_dir} does not exist.")
else:
    try:
        # Load the Pegasus model and tokenizer
        tokenizer = PegasusTokenizer.from_pretrained(model_dir)
        model = PegasusForConditionalGeneration.from_pretrained(model_dir)
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
    else:
        @st.cache_data(hash_funcs={type(tokenizer): id})
        def text_summary(text):
            # Tokenize the input text
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="longest")

            # Generate the summary
            summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, num_beams=4, early_stopping=True)

            # Decode the summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary

        def extract_text_from_pdf(file_path):
            # Open the PDF file using PyPDF2
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                page = reader.pages[0]
                text = page.extract_text()
            return text

        choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document"])

        if choice == "Summarize Text":
            st.subheader("Summarize Text")
            input_text = st.text_area("Enter your text here")
            if input_text:
                if st.button("Summarize Text"):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("**Your Input Text**")
                        st.info(input_text)
                    with col2:
                        st.markdown("**Summary Result**")
                        result = text_summary(input_text)
                        st.success(result)

        elif choice == "Summarize Document":
            st.subheader("Summarize Document using Pegasus")
            input_file = st.file_uploader("Upload your document here", type=['pdf'])
            if input_file:
                if st.button("Summarize Document"):
                    with open("doc_file.pdf", "wb") as f:
                        f.write(input_file.getbuffer())
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.info("File uploaded successfully")
                        extracted_text = extract_text_from_pdf("doc_file.pdf")
                        st.markdown("**Extracted Text is Below:**")
                        st.info(extracted_text)
                    with col2:
                        st.markdown("**Summary Result**")
                        doc_summary = text_summary(extracted_text)
                        st.success(doc_summary)
