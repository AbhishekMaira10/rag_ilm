from fastapi import FastAPI
from ragatouille import RAGPretrainedModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI(title="nanonets API", version="0.1.1")

@app.get("/hi")
async def hi():
	return {"message": "Hi"}