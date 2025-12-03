from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader, WebBaseLoader, CSVLoader

# loading single plain text file
# loader = TextLoader(file_path="DocumentLoaders/data/data.txt", encoding="utf-8")
# documents = loader.load()
# print(documents)
#-------------------------------------------------------------------------------
##load single pdf text file
# loader = PyPDFLoader(file_path="DocumentLoaders/data/sample.pdf")
# documents = loader.load()
# print(documents)
#-------------------------------------------------------------------------------
# ## loading multiple files from dir
# txt_loader = DirectoryLoader(path="DocumentLoaders/data/", glob="**/*.txt", loader_cls=TextLoader)
# pdf_loader = DirectoryLoader(path="DocumentLoaders/data/", glob="**/*.pdf", loader_cls=PyPDFLoader)
# docs = TextLoader.load(txt_loader) + PyPDFLoader.load(pdf_loader)
# print(docs)
#-------------------------------------------------------------------------------
## web-based loader
# url = ["https://www.example.com/", "https://google.com"]
# loader = WebBaseLoader(url)
# documents = [i for i in loader.lazy_load()]
# print(documents)

## csv loader
loader = CSVLoader(file_path="DocumentLoaders/data/t0.csv")
documents = loader.load()
print(documents)

