from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()
template1 = PromptTemplate(
    input_variables=["city"],
    template="Provide a brief description of the city {city}'s tourist spots in less than 100 words. "
             "Respond only with the description without any additional text.",
)
template2 = PromptTemplate(
    template="Bellow is the description of a tourist spots in the {city} city. Pick top 3 must to try :\n{description}",
    input_variables=["city","description"],
)

chain = template1 | model | parser | template2 | model | parser
response = chain.invoke({'city':'bangalore'})
print(response)
