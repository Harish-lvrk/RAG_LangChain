# Understanding Semantic Meaning-Based Text Splitting in LangChain
# Purpose: This script demonstrates how to split text based on semantic meaning
# using embeddings to identify natural topic boundaries in the text

# Import required components
from langchain_experimental.text_splitter import SemanticChunker     # Advanced semantic splitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings      # For generating text embeddings
from dotenv import load_dotenv                                       # For API key management

# Load environment variables (API keys)
load_dotenv()

# STEP 1: Initialize the Semantic Chunker
# SemanticChunker uses embeddings to understand meaning and find natural break points
text_splitter = SemanticChunker(
    # Use Google's Gemini embeddings model
    GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    # How to determine break points:
    breakpoint_threshold_type="standard_deviation",  # Use statistical measure
    breakpoint_threshold_amount=1                   # Sensitivity of the split
)

# STEP 2: Prepare sample text with distinct topics
# This sample deliberately contains three different topics:
# 1. Farming and nature
# 2. Cricket and sports
# 3. Terrorism and security
sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

# STEP 3: Split the text based on semantic meaning
# create_documents() will analyze the text and split at semantic boundaries
docs = text_splitter.create_documents([sample])

# Display results
print(f"Number of semantic chunks: {len(docs)}")
print("\nGenerated chunks with semantic boundaries:")
print(docs)

# Educational Notes:
# 1. Semantic Splitting Benefits:
#    - Splits text based on meaning, not just structure
#    - Preserves topic coherence
#    - Better for content understanding
#
# 2. How It Works:
#    - Converts text into embeddings (numerical representations)
#    - Measures semantic similarity between sections
#    - Identifies natural topic boundaries
#    - Splits where topics change significantly
#
# 3. Use Cases:
#    - Content categorization
#    - Topic segmentation
#    - Intelligent document summarization
#    - Question-answering systems
#
# 4. Considerations:
#    - More computationally intensive than other splitters
#    - Requires embedding model access
#    - May need tuning based on content type

