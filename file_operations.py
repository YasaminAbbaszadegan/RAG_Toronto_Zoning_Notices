# file_operations.py
import os
import shutil

def cache_uploaded_documents(uploaded_files, upload_folder):
    """
    Caches uploaded documents to a specified folder.

    Args:
        uploaded_files (list): List of uploaded files.
        upload_folder (str): Path to the folder where documents will be cached.

    Returns:
        list: List of file paths of the cached documents.
    """
    cached_documents = []
    for uploaded_file in uploaded_files:
        folder_path = os.path.join(upload_folder, uploaded_file.name)
        with open(folder_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        cached_documents.append(folder_path)
    return cached_documents

def remove_uploaded_folder(upload_folder):
    """
    Removes the uploaded folder directory.

    Args:
        upload_folder (str): Path to the uploaded folder.
    """
    if os.path.exists(upload_folder):
        shutil.rmtree(upload_folder)
