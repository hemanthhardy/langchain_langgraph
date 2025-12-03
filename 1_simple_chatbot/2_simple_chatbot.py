from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_history = []
chat_history.append(SystemMessage(content="You are an AI assistant, help answer the user queries"))

user_input = input("User : ")
while user_input != 'exit':
    chat_history.append(HumanMessage(content=user_input))
    # print('-------------\n',chat_history,'\n----------\n')
    ai_response = model.invoke(chat_history)
    print(ai_response.content)
    chat_history.append(AIMessage(content=ai_response.content))
    user_input = input("User : ")
print('Chat Exited...')