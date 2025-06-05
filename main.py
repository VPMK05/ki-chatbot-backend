from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import fitz
import openai
import os
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")
pdf_texts = []

class ChatRequest(BaseModel):
    message: str

@app.post("/uploadpdf/")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    doc = fitz.open(stream=content, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    pdf_texts.append(full_text)
    return {"filename": file.filename, "length": len(full_text)}

@app.post("/chat/")
async def chat(request: ChatRequest):
    context = "\n".join(pdf_texts[-3:]) if pdf_texts else ""
    prompt = f"Kontext:\n{context}\n\nFrage: {request.message}"
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7,
    )
    return {"answer": response.choices[0].message.content}
