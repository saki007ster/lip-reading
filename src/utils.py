import os
import requests
import streamlit as st

def download_file(url, destination, description="Downloading model..."):
    """
    Downloads a file from a URL to a destination path.
    If run within a Streamlit app, it shows a progress bar.
    """
    if os.path.exists(destination):
        return True

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"{description} (0%)")
        
        downloaded_size = 0
        chunk_size = 1024 * 1024 # 1MB
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = min(1.0, downloaded_size / total_size)
                        progress_bar.progress(progress)
                        status_text.text(f"{description} ({int(progress * 100)}%)")
        
        progress_bar.empty()
        status_text.empty()
        return True
    except Exception as e:
        if os.path.exists(destination):
            os.remove(destination)
        st.error(f"Error downloading {description}: {e}")
        return False
