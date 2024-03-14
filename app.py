# main.py
import os
import streamlit as st
import pandas as pd
from initialize_llama import initialize_llama
from file_operations import cache_uploaded_documents, remove_uploaded_folder

st.set_page_config(page_icon="./icon/Toronto-logo-blue-750.jpg")
st.image("./icon/Toronto-logo-blue-750.jpg", use_column_width=True, width=30)
st.title('Zoning Decision Assistant: City Notice Reader')

upload_folder = 'temp'
if upload_folder and not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

uploaded_files = st.file_uploader("Upload multiple documents", accept_multiple_files=True)
first_button_run_finished = False

if uploaded_files and upload_folder:
    cached_documents = cache_uploaded_documents(uploaded_files, upload_folder)
    document_paths = [f"./{upload_folder}/" + file_content for file_content in os.listdir(f'./{upload_folder}/')]
    sentence_window_engines = initialize_llama(document_paths)
    query = st.text_input("What would you like to ask?")

    if st.button("Submit Query"):
        if not query.strip():
            st.error(f"Please provide the search query.")
        else:
            try:
                combined_responses = {}
                for doc_title, engine in sentence_window_engines.items():
                    response = engine.query(query)
                    combined_responses[doc_title] = response.response

                st.write(combined_responses)
                df = pd.DataFrame.from_dict(combined_responses, orient='index', columns=['Response'])
                st.dataframe(df)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    first_button_run_finished = True

if first_button_run_finished:
    if st.button("Remove Uploaded Folder"):
        remove_uploaded_folder(upload_folder)
