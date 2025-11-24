from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

str_parser = StrOutputParser()
class review_sentiment(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(description='The overall sentiment of the review')

pydantic_parser = PydanticOutputParser(pydantic_object=review_sentiment)

template1 = PromptTemplate(
    template = """Analyze the sentiment of the following movie review and classify it as positive, negative, or neutral:\n\n {review_text}\n\n
    Provide only the sentiment in your response in the following format {response_format}.""",
    input_variables = ['review_text'],
    partial_variables ={"response_format":pydantic_parser.get_format_instructions()}
)

chain = template1 | model | pydantic_parser 
response1 = chain.invoke({'review_text': "The product is very good and I am satisfied with my purchase."})
print(response1)

template2 = PromptTemplate(
    template = """give an welcoming reply to the reviewer in less than 50 words for the the following positive review:\n\n {review_text}\n\n""",
    input_variables = ['review_text'])

template3 = PromptTemplate(
    template = """give a sorry reply and ask for the in details reasons specifically to improve our product to the reviewer in less than 50 words for the the following negative review:\n\n {review_text}\n\n""",
    input_variables = ['review_text'])

branching_chain = chain | RunnableBranch(
    (lambda x: x.sentiment == 'positive', template2 | model | str_parser),
    (lambda x: x.sentiment == 'negative', template3 | model | str_parser),
    RunnableLambda(lambda x: "Thank you for your feedback!")
) | str_parser

# print(branching_chain.invoke({'review_text': "The product is okay, not too bad but not great either."}))
# print(branching_chain.invoke({'review_text': "The product is great."}))
print(branching_chain.invoke({'review_text': "The product is terrible and broke after one use."}))

