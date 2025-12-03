from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import CharacterTextSplitter , RecursiveCharacterTextSplitter

loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# character splitter- ------------------------------------------------------------ Splits at exact character count
# text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
# texts = text_splitter.split_documents(documents)
## texts = text_splitter.split_text(documents)  #-- for plain text like: documents = '<text content>'
# # print(len(texts))
# print(texts[4])


# recursive character splitter- --------------------------------------------------- Finds natural break points
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = text_splitter.split_documents(documents)
# print(len(texts))
print(texts[4])
