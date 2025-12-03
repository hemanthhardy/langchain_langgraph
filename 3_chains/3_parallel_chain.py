from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv(override=True)

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
str_parser = StrOutputParser()
template1 = PromptTemplate(
    template = """Procide the cast name, studio name, creator and owner of the superhero character:\n\n {superhero_character1}\n""",
    inpt_variables = ['superhero_character1']
)
template2 = PromptTemplate(
    template = """Provide the main superpowers and nemesis of the superhero character in the movie:\n\n{mvoie_name}""",
    input_variables = ['mvoie_name']
)
template3 = PromptTemplate(
    template = """Check whether the two statements are Factual and corelation:\n\n statement-1: {statement_1}\n\n statement-2: {statement_2}\n\n""",
    input_variables = ['statement_1','statement_2']
)
str_parser  = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'statement_1': template1 | model | str_parser,
        'statement_2': template2 | model | str_parser,
    })

full_chain = parallel_chain | template3 | model | str_parser

result = full_chain.invoke(
    {
        'superhero_character1': 'Peter Parker',
        'mvoie_name': 'Iron man'
    }
)
# result = full_chain.invoke(
#     {
#         'superhero_character1': 'Tony Stark',
#         'mvoie_name': 'Iron man'
#     }
# )

print(result)