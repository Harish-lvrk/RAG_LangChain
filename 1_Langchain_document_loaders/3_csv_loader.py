# Understanding CSV Data Loading in LangChain
# Purpose: This script demonstrates how to load structured data from CSV files
# CSV (Comma-Separated Values) files are common for tabular data like spreadsheets or databases

# Import the CSV loader from LangChain
from langchain_community.document_loaders import CSVLoader  # Specialized loader for CSV files

# STEP 1: Initialize the CSV loader
# CSVLoader can handle:
# - Different delimiters (comma, tab, etc.)
# - Headers and no headers
# - Custom column selection
# - Row-based document creation
loader = CSVLoader(file_path='Social_Network_Ads.csv')

# STEP 2: Load the CSV content
# The load() method:
# - Reads the CSV file
# - Converts each row into a Document object
# - Preserves column headers as metadata
docs = loader.load()

# STEP 3: Analyze the loaded content
# Print number of rows (each doc represents one row)
print(f"Number of rows in CSV: {len(docs)}")

# Print the second row's content and metadata
print("\nSecond row data:")
print(docs[1])

# Educational Notes:
# 1. Row-based Documents: Each row becomes a separate Document
# 2. Metadata Handling: Column headers are preserved in metadata
# 3. Data Structure: CSVLoader maintains the tabular structure
# 4. Use Cases:
#    - Loading structured datasets for analysis
#    - Processing tabular data for ML/AI tasks
#    - Converting spreadsheet data into document format