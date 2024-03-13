import os
import shutil
import streamlit as st
import pandas as pd
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.postprocessor import SentenceTransformerRerank, MetadataReplacementPostProcessor

# Function to initialize LLAMA model and necessary components
def initialize_llama(documents_paths):
    # Dictionary to store query engines for each document
    sentence_window_engines = {}

    # Initialize LLAMA model
    llm = LlamaCPP(model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf', temperature=1)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    node_parser = SentenceWindowNodeParser.from_defaults(window_size=3, window_metadata_key="window", original_text_metadata_key="original_text")
    sentence_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=node_parser)
    rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base")
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    
    # Iterate through each document path
    for document_path in documents_paths:
        # Extract title from the filename
        title = os.path.splitext(os.path.basename(document_path))[0]
        
        # Load document content
        documents = SimpleDirectoryReader(input_files=[document_path]).load_data()
        documents = Document(text="\n\n".join([doc.text for doc in documents]))

        # Create VectorStoreIndex and query engine
        sentence_index = VectorStoreIndex.from_documents([documents], service_context=sentence_context)
        sentence_window_engine = sentence_index.as_query_engine(similarity_top_k=6, node_postprocessors=[postproc, rerank])
        
        # Store query engine for the document
        sentence_window_engines[title] = sentence_window_engine

    return sentence_window_engines

# Function to cache uploaded documents
def cache_uploaded_documents(uploaded_files, upload_folder):
    cached_documents = []
    # Iterate through each uploaded file
    for uploaded_file in uploaded_files:
        folder_path = os.path.join(upload_folder, uploaded_file.name)
        # Write file content to the specified folder
        with open(folder_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        cached_documents.append(folder_path)
    return cached_documents

# Function to remove the uploaded folder directory
def remove_uploaded_folder(upload_folder):
    if os.path.exists(upload_folder):
        shutil.rmtree(upload_folder)

# Set Streamlit app configuration and title
st.set_page_config(page_title="Notice of Decision Zoning Toronto Document Reader", page_icon="./icon/Toronto-logo-blue-750.jpg")
st.image("./icon/Toronto-logo-blue-750.jpg", use_column_width=True, width=30)
st.title('Notice of Decision Zoning Toronto Document Reader')

# Define upload folder and create if it doesn't exist
upload_folder='temp'
if upload_folder and not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# File uploader for multiple documents
uploaded_files = st.file_uploader("Upload multiple documents", accept_multiple_files=True)
first_button_run_finished = False

# If files are uploaded and folder name is specified
if uploaded_files and upload_folder:
    # Cache the uploaded documents
    cached_documents = cache_uploaded_documents(uploaded_files, upload_folder)
    # Get paths of the uploaded documents
    document_paths = [f"./{upload_folder}/" + file_content for file_content in os.listdir(f'./{upload_folder}/')]
    # Initialize the LLAMA model and query engines
    sentence_window_engines = initialize_llama(document_paths)
    query = st.text_input("What would you like to ask?")

    # If 'Submit Query' button is clicked
    if st.button("Submit Query"):
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

# Show "Remove Uploaded Folder" butt
if first_button_run_finished:
    if st.button("Remove Uploaded Folder"):
        remove_uploaded_folder(upload_folder)