# Understanding Text Loaders in LangChain
# Purpose: This script demonstrates how to load and process text files using LangChain's TextLoader
# and how to use the loaded content with language models for text analysis.

# Import necessary libraries
from langchain_community.document_loaders import TextLoader  # For loading text files
from langchain_google_genai import ChatGoogleGenerativeAI   # Google's Gemini model interface
from langchain_core.output_parsers import StrOutputParser   # Parse model output to string
from langchain_core.prompts import PromptTemplate           # Template for structuring prompts
from dotenv import load_dotenv                             # Load environment variables

# Load environment variables (like API keys) from .env file
load_dotenv()

# Initialize the Gemini model
# Note: We use gemini-2.5-flash for faster processing while maintaining good quality
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Create a prompt template for poem summary
# The {poem} placeholder will be replaced with actual content
prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

# Initialize the output parser to convert model output to string
parser = StrOutputParser()

# STEP 1: Load the text file
# TextLoader is used for simple text files (.txt)
# encoding='utf-8' ensures proper handling of special characters
loader = TextLoader('1_Langchain_document_loaders/cricket.txt', encoding='utf-8')

# STEP 2: Load the content
# The load() method returns a list of Document objects
docs = loader.load()

# Understanding the loaded document
print("Type of loaded document:", type(docs))  # Should be a list of Documents
print("Number of documents:", len(docs))       # How many documents were loaded
print("\nDocument content:")
print(docs[0].page_content)                   # The actual text content
print("\nDocument metadata:")
print(docs[0].metadata)                       # File information like path, hash, etc.

# STEP 3: Process the text using our language model chain
# Chain composition: prompt -> model -> parser
chain = prompt | model | parser

# Generate and print the summary
print("\nGenerated Summary:")
print(chain.invoke({'poem':docs[0].page_content}))

# Educational Notes:
# 1. Document Objects: LangChain uses Document objects that contain both content
#    and metadata, making it easy to track source information.
# 2. Chain Composition: The | operator creates a processing pipeline, making it
#    clear how data flows through the system.
# 3. Async Support: LangChain supports async operations for better performance
#    in production environments.

