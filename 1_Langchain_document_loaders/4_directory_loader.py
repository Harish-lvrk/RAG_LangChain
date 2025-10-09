# Understanding Directory Loading in LangChain
# Purpose: This script demonstrates how to load multiple files from a directory
# using LangChain's DirectoryLoader, which is useful for batch processing

# Import necessary loaders
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader  # For loading multiple files

# STEP 1: Initialize the Directory loader
# DirectoryLoader parameters:
# - path: Directory to search for files
# - glob: Pattern to match files (e.g., '*.pdf' for all PDF files)
# - loader_cls: The loader class to use for individual files
loader = DirectoryLoader(
    path='books',           # Directory containing the files
    glob='*.pdf',          # Only load PDF files
    loader_cls=PyPDFLoader # Use PyPDFLoader for each file
)

# STEP 2: Load the documents using lazy loading
# lazy_load() advantages:
# - Memory efficient: Loads files one at a time
# - Better for large directories
# - Allows processing to start before all files are loaded
docs = loader.lazy_load()

# STEP 3: Process each document
# Iterate through the loaded documents and print their metadata
print("Processing documents in directory:")
for document in docs:
    print(document.metadata)

# Educational Notes:
# 1. Batch Processing: DirectoryLoader helps process multiple files efficiently
# 2. Pattern Matching: Use glob patterns to filter files:
#    - '*.pdf' for PDFs
#    - '**/*.txt' for recursive text file search
#    - '*.*' for all files
# 3. Memory Efficiency:
#    - lazy_load() for memory-efficient processing
#    - load() for immediate loading of all files
# 4. Common Use Cases:
#    - Processing document collections
#    - Batch importing data
#    - Building document databases