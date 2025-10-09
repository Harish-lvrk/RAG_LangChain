from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('2_langchain_text_splitters/Sebastian Raschka, Yuxi (Hayden) Liu, Vahid Mirjalili - Machine Learning with PyTorch and Scikit-Learn_ Develop machine learning and deep learning models with Python (2022, Packt Publishing) - libgen.li.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print(result[1].page_content)