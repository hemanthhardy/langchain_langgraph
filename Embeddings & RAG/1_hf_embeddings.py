from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
documents = [
    "The Eiffel Tower is located in Paris.",
    "The capital of Germany is Berlin.",
    "The Great Wall of China is a historic landmark.",
    "The capital of France is Delhi.",
    "Mount Everest is the highest mountain in the world.",
    "The capital of Italy is Rome." 
]
query = "What is the capital of France?"
query_embedding = embedding_model.embed_query(query)
doc_embeddings = embedding_model.embed_documents(documents)
# print(doc_embeddings)
similarities = cosine_similarity([query_embedding], doc_embeddings)
print(similarities)
most_similar_doc_index = similarities.argmax()
most_similar_document = documents[most_similar_doc_index]
print("Most similar document:", most_similar_document)

query = query +'\nrefer the below data to answer:\n'+  most_similar_document
print('\n\nModel Input: \n', query,'\n')
result =  model.invoke(query)
# print(result.content)
print(result)
