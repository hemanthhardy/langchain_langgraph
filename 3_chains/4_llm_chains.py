from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import LLMChain

load_dotenv(override=True)

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
str_parser = StrOutputParser()

template = PromptTemplate(
    template = """Provide a brief review of the following movie in less than 50 words:\n\n {movie_name}\n\n""",
    input_variables = ['movie_name']
)
chain = LLMChain(
    prompt = template,
    llm = model,
    output_parser = str_parser
)
result = chain.invoke(
    {
        'movie_name': 'Inception'
    }
)
print(result)