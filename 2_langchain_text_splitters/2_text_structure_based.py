# Understanding Text Structure-Based Splitting in LangChain
# Purpose: This script demonstrates how to split text while preserving its natural structure
# using RecursiveCharacterTextSplitter, which respects paragraph breaks and sentences

# Import the recursive text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# STEP 1: Define sample text with natural structure
# Note how the text has:
# - Multiple paragraphs
# - Natural sentence breaks
# - Varied punctuation
text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what's possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

# STEP 2: Initialize the RecursiveCharacterTextSplitter
# This splitter tries to split on (in order):
# 1. Line breaks (\n\n)
# 2. Paragraph breaks
# 3. Sentences
# 4. Words
# Only if no higher-level splits are possible will it resort to character-level splits
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Maximum size of each chunk
    chunk_overlap=0,     # No overlap between chunks
)

# STEP 3: Split the text
# split_text() method processes the text and returns chunks
chunks = splitter.split_text(text)

# Display results
print(f"Number of chunks created: {len(chunks)}")
print("\nGenerated chunks:")
print(chunks)

# Educational Notes:
# 1. Recursive Splitting Strategy:
#    - Tries to split at natural boundaries first
#    - Falls back to less ideal splits if necessary
#    - Preserves context better than simple character splitting
#
# 2. When to Use RecursiveCharacterTextSplitter:
#    - Working with formatted text
#    - Preserving paragraph structure is important
#    - Need to maintain sentence coherence
#
# 3. Advantages:
#    - More natural text divisions
#    - Better context preservation
#    - Improved readability of chunks
#
# 4. Common Parameters:
#    - separators: List of separators to try
#    - chunk_size: Maximum chunk size
#    - chunk_overlap: Overlap between chunks
#    - length_function: Function to measure text length_splitter import RecursiveCharacterTextSplitter

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)