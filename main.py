import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import psycopg2
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your Vercel domain once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
COLLECTION = "explore_pakistan_destinations"

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


class ChatRequest(BaseModel):
    query: str
    session_id: str


def get_embedding(text: str):
    return embed_model.encode(text).tolist()


def save_message(session_id: str, role: str, content: str):
    conn = psycopg2.connect(os.getenv("NEON_DATABASE_URL"), sslmode="require")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_history (session_id, role, content) VALUES (%s, %s, %s)",
        (session_id, role, content),
    )
    conn.commit()
    cur.close()
    conn.close()


@app.get("/")
async def root():
    return {"status": "Explore Pakistan chatbot API is running"}


@app.post("/chat")
async def chat(req: ChatRequest):
    # 1. Embed the query and search Qdrant for relevant destination content
    query_vector = get_embedding(req.query)
    results = qdrant_client.query_points(
        collection_name=COLLECTION, query=query_vector, limit=3
    )
    hits = results.points
    context = "\n\n".join([hit.payload["text"] for hit in hits]) if hits else ""

    # 2. Generate a response grounded in that context, using Groq (free)
    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful travel assistant for Explore Pakistan. "
                    "Answer ONLY using the provided context about destinations. "
                    "If the answer isn't in the context, say you don't have "
                    "that information yet, but suggest checking the "
                    "Destinations page. Keep answers concise and friendly."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {req.query}",
            },
        ],
    )
    answer = completion.choices[0].message.content

    # 3. Save to Postgres
    save_message(req.session_id, "user", req.query)
    save_message(req.session_id, "assistant", answer)

    # 4. Return response
    return {"response": answer}