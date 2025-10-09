# Understanding PDF Document Loading in LangChain
# Purpose: This script demonstrates how to load and process PDF documents using LangChain's PyPDFLoader
# PDF files are more complex than text files and require special handling for proper extraction.

# Import the PDF loader from LangChain
from langchain_community.document_loaders import PyPDFLoader  # Specialized loader for PDF files

# STEP 1: Initialize the PDF loader
# PyPDFLoader automatically handles:
# - PDF parsing
# - Text extraction
# - Page segmentation
loader = PyPDFLoader('dl-curriculum.pdf')

# STEP 2: Load the PDF content
# The load() method:
# - Splits the PDF into pages
# - Extracts text from each page
# - Creates a separate Document object for each page
docs = loader.load()

# STEP 3: Analyze the loaded content
# Print number of pages (each doc represents one page)
print(f"Number of pages in the PDF: {len(docs)}")

# Print content of the first page
print("\nContent of first page:")
print(docs[0].page_content)

# Print metadata of the second page
# Metadata includes important information like:
# - page number
# - source file
# - any PDF-specific metadata
print("\nMetadata of second page:")
print(docs[1].metadata)

# Educational Notes:
# 1. Page-based Documents: Unlike text files, PDFs are split into pages automatically
# 2. OCR Consideration: PyPDFLoader works best with machine-readable PDFs
#    For scanned documents, consider using other loaders with OCR capabilities
# 3. Metadata: PDF metadata is preserved, helping track source pages and document properties
# 4. Memory Usage: Large PDFs are handled efficiently by processing one page at a time