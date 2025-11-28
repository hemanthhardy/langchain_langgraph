from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv(override=True)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt1  = PromptTemplate(template = "Generate a tweet about {topic}", input_variables=["topic"])
prompt2  = PromptTemplate(template = "Generate the linkedin post about:\n {topic}", input_variables=["topic"])

parser_str = StrOutputParser()
parallel_chain = RunnableParallel(
    {
        'tweet': RunnableSequence(prompt1, model, parser_str),
        'linkedin post': RunnableSequence(prompt2, model, parser_str)
    }
)

result = parallel_chain.invoke({"topic":"agentic ai"})
print(result)
print('\n\nTweet :',result['tweet'])
print('\n\nLinkedin Post :',result['linkedin post'])