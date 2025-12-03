from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_template = ChatPromptTemplate(
    [
        ('system',"you are a helpful {domain} Expert. And help users interview preperations, by gathering their informations. Keep your answers less than 50 words"),
        MessagesPlaceholder(variable_name = "chat_history"),
        ('human','{query}')
    ]
)

chat_history = []
with open('simple_chat_history.txt','r') as file:
    file = file.readlines()
    for line in file:
        print(line)
# print(i for line in file)
user_input = input("User : ")
while user_input != 'exit':

    with open('simple_chat_history.txt','r') as file:
        chat_history.extend(file.readlines())

    prompt = chat_template.invoke({
        'domain':'Computer Science and Engineering',
        'chat_history':chat_history,
        'query': user_input
    })
    ai_response = model.invoke(prompt)
    print(ai_response.content)
    with open('simple_chat_history.txt','a') as file:
        file.write('\n'.join([f"HumanMessage(content={user_input})",f"AIMessage(content={ai_response.content})\n"]))
    user_input = input("User : ")
print('Chat Exited...')