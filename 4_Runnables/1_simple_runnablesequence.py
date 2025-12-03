from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv(override=True)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt1  = PromptTemplate(template = "Tell me a thirukural in tamil about {topic} ", input_variables=["topic"])
prompt2  = PromptTemplate(template = "Translate the below tamil thirukural to english:\n {thirukural} ", input_variables=["thirukural"])

runnable = RunnableSequence(prompt1,model,prompt2,model,StrOutputParser())

result = runnable.invoke({"topic":"topic of friendship"})
print(result)