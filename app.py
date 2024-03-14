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

# Display file uploader for multiple documents
uploaded_files = st.file_uploader("Upload multiple documents", accept_multiple_files=True)
first_button_run_finished = False

# If files are uploaded and upload folder is specified
if uploaded_files and upload_folder:
    # Cache the uploaded documents to the specified folder
    cached_documents = cache_uploaded_documents(uploaded_files, upload_folder)
    # Get paths of the uploaded documents
    document_paths = [f"./{upload_folder}/" + file_content for file_content in os.listdir(f'./{upload_folder}/')]
    
    # Display buttons for selecting node parser options
    parser_option = st.radio("Select Node Parser Option:", ("Basic Parser", "Sentence Window Parser"))

    # Set flag based on user's choice of node parser
    use_advanced_node_parser = True if parser_option == "Sentence Window Parser" else False

    # Initialize the LLAMA model and query engines for the uploaded documents
    sentence_window_engines = initialize_llama(document_paths, use_advanced_node_parser)
    # Get user input query
    query = st.text_input("What would you like to ask?")

    # If 'Submit Query' button is clicked
    if st.button("Submit Query"):
        # Check if query is provided
        if not query.strip():
            st.error(f"Please provide the search query.")
        else:
            try:
                combined_responses = {}
                # Query each document using corresponding query engine
                for doc_title, engine in sentence_window_engines.items():
                    response = engine.query(query)
                    combined_responses[doc_title] = response.response

                # Display the combined responses
                st.write(combined_responses)
                # Convert responses to DataFrame and display
                df = pd.DataFrame.from_dict(combined_responses, orient='index', columns=['Response'])
                st.dataframe(df)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Update flag to indicate the first button run is finished
    first_button_run_finished = True

# If the first button run is finished, show the "Remove Uploaded Folder" button
if first_button_run_finished:
    if st.button("Remove Uploaded Folder"):
        # Remove the uploaded folder directory
        remove_uploaded_folder(upload_folder)
