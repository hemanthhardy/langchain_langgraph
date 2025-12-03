from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class MovieReview(TypedDict):
    sentiment:str
    rating:int
    summary: str

model = model.with_structured_output(MovieReview)

prompt = 'The Silent Horizon is a visually stunning and emotionally resonant drama that blends breathtaking cinematography with a deeply human story. The pacing is deliberate, allowing the audience to fully absorb the characters’ struggles and triumphs. While a few scenes feel slightly drawn out, the film’s heartfelt performances and haunting score make it a memorable watch. Perfect for those who enjoy slow-burn storytelling with a powerful emotional payoff.'
# prompt = '"The Dark Knight Rises": Ebert notes that the film suffers from self-importance and convoluted storytelling, ultimately failing to deliver a satisfying conclusion to the trilogy.'
result = model.invoke(prompt)
print(result)