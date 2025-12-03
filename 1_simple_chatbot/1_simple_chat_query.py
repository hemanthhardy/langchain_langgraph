from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
query = "what is the capital city of India"
result =  model.invoke(query)
print(result.content)
print(result)