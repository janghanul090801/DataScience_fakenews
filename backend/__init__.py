from fastapi import FastAPI
from pydantic import BaseModel
from backend.model import LR, vectorizer

app = FastAPI()

class NewsTextRequest(BaseModel):
    text: str

class IsFakenewsResponse(BaseModel):
    is_fake: bool

@app.post('/')
async def jinjjallueayogajjallueayo(request: NewsTextRequest) -> IsFakenewsResponse:
    result = LR.predict(vectorizer.transform([request.text]))
    return IsFakenewsResponse(is_fake=(result[0] == 0))