from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

load_dotenv(override=True)

documents = TextLoader("data.txt").load()
documents = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(documents)
print(len(documents))
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
qa_chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=retriever, return_source_documents=True)

query = "summarise the purpose section of the document"
result = qa_chain.invoke(query)
print("Answer:", result['result'])
print("Source Documents:", result['source_documents'])
print('\n\n\n',result)
