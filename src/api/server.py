from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routers import chat, documents

app = FastAPI(title="Research Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)

@app.get("/")
def read_root():
    return {"message": "API is running."}
