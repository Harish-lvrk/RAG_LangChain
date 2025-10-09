# Understanding Length-Based Text Splitting in LangChain
# Purpose: This script demonstrates how to split large documents into smaller chunks
# based on character length, which is crucial for working with LLMs that have token limits

# Import necessary components
from langchain.text_splitter import CharacterTextSplitter    # For splitting text by length
from langchain_community.document_loaders import PyPDFLoader  # For loading PDF files

# STEP 1: Load the source document
# Load a machine learning book PDF for demonstration
loader = PyPDFLoader('2_langchain_text_splitters/Sebastian Raschka, Yuxi (Hayden) Liu, Vahid Mirjalili - Machine Learning with PyTorch and Scikit-Learn_ Develop machine learning and deep learning models with Python (2022, Packt Publishing) - libgen.li.pdf')

# Load all pages from the PDF
docs = loader.load()

# STEP 2: Configure the text splitter
# CharacterTextSplitter parameters:
# - chunk_size: Maximum number of characters in each chunk
# - chunk_overlap: Number of characters to overlap between chunks (helps maintain context)
# - separator: Character to use for splitting (empty string means split at any character)
splitter = CharacterTextSplitter(
    chunk_size=200,     # Each chunk will be ~200 characters
    chunk_overlap=0,    # No overlap between chunks
    separator=''        # Split at any character position
)

# STEP 3: Split the documents into chunks
# split_documents() preserves metadata from original documents
result = splitter.split_documents(docs)

# Print the second chunk to see the result
print("Sample chunk from the split document:")
print(result[1].page_content)

# Educational Notes:
# 1. Why Split Text?
#    - LLMs have token limits (e.g., 4096 tokens for GPT-3.5)
#    - Smaller chunks are easier to process
#    - Enables more precise information retrieval
#
# 2. Chunk Size Considerations:
#    - Too small: May lose context and coherence
#    - Too large: May exceed model's token limit
#    - Ideal size depends on your use case
#
# 3. Overlap Benefits:
#    - Helps maintain context between chunks
#    - Prevents cutting in the middle of important information
#    - Typical overlap: 10-20% of chunk size
#
# 4. Use Cases:
#    - Processing long documents for QA systems
#    - Creating document embeddings
#    - Building knowledge bases