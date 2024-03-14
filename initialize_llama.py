# initialize_llama.py
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter

import os
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.postprocessor import SentenceTransformerRerank, MetadataReplacementPostProcessor

# def initialize_llama(documents_paths):
#     """
#     Initializes the LLAMA model and creates query engines for each document.

#     Args:
#         documents_paths (list): List of file paths of the documents to be indexed.

#     Returns:
#         dict: Dictionary containing document titles as keys and corresponding query engines as values.
#     """
#     sentence_window_engines = {}
#     llm = LlamaCPP(model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf', 
#                    temperature=1, 
#                    model_kwargs={"n_gpu_layers": -1},
#                    verbose=True,)
#     embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
#     node_parser = SentenceWindowNodeParser.from_defaults(window_size=3, window_metadata_key="window", original_text_metadata_key="original_text")
#     sentence_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=node_parser)
#     rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base")
#     postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    
#     for document_path in documents_paths:
#         title = os.path.splitext(os.path.basename(document_path))[0]
#         documents = SimpleDirectoryReader(input_files=[document_path]).load_data()
#         documents = Document(text="\n\n".join([doc.text for doc in documents]))
#         sentence_index = VectorStoreIndex.from_documents([documents], service_context=sentence_context)
#         sentence_window_engine = sentence_index.as_query_engine(similarity_top_k=6, node_postprocessors=[postproc, rerank])
#         sentence_window_engines[title] = sentence_window_engine

#     return sentence_window_engines



# Function to initialize LLAMA model and necessary components
def initialize_llama(documents_paths, use_advanced_node_parser):
    """
    Initializes the LLAMA model and creates query engines for each document.

    Args:
        documents_paths (list): List of file paths of the documents to be indexed.
        use_advanced_node_parser (bool): Flag indicating whether to use advanced node parser or not.

    Returns:
        dict: Dictionary containing document titles as keys and corresponding query engines as values.
    """

    sentence_window_engines = {}
    llm = LlamaCPP(model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf', 
                   temperature=1, 
                   model_kwargs={"n_gpu_layers": -1},
                   verbose=True,)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Select node parser based on user's choice
    if use_advanced_node_parser:
        node_parser = SentenceWindowNodeParser.from_defaults(window_size=10, window_metadata_key="window", original_text_metadata_key="original_text")
    else:
        node_parser = SentenceSplitter() #chunk_size=500, chunk_overlap=20
    
    sentence_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=node_parser)
    rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base")
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    
    for document_path in documents_paths:
        title = os.path.splitext(os.path.basename(document_path))[0]
        documents = SimpleDirectoryReader(input_files=[document_path]).load_data()
        documents = Document(text="\n\n".join([doc.text for doc in documents]))
        sentence_index = VectorStoreIndex.from_documents([documents], service_context=sentence_context)
        sentence_window_engine = sentence_index.as_query_engine(similarity_top_k=6, node_postprocessors=[postproc, rerank])
        sentence_window_engines[title] = sentence_window_engine

    return sentence_window_engines