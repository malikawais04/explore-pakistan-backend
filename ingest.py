import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
COLLECTION = "explore_pakistan_destinations"

# Free local embedding model — downloads once (~80MB), then runs on your machine
print("Loading embedding model (first run downloads it, may take a minute)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_SIZE = 384  # this model outputs 384-dimensional vectors

destinations = [
    {
        "slug": "hunza-valley",
        "title": "Hunza Valley",
        "text": (
            "Hunza Valley is one of Pakistan's most iconic destinations, known for "
            "its dramatic mountain scenery, centuries-old forts, and warm hospitality "
            "of the Hunzakut people. Home to Rakaposhi and Ultar Sar peaks, the valley "
            "offers stunning views year-round, from blossoming apricot orchards in "
            "spring to golden foliage in autumn. Best time to visit: March to October. "
            "How to reach: Fly to Gilgit then a 2-hour drive, or drive the full "
            "Karakoram Highway from Islamabad (18-20 hours). Highlights include "
            "Baltit Fort, Attabad Lake, Passu Cones, and the Eagle's Nest viewpoint."
        ),
    },
    {
        "slug": "skardu",
        "title": "Skardu",
        "text": (
            "Skardu is the launch point for expeditions to K2 and other giants of the "
            "Karakoram range. Beyond mountaineering, it offers serene lakes, desert "
            "landscapes at high altitude, and access to Deosai National Park, one of "
            "the highest plateaus in the world. Best time to visit: April to September. "
            "How to reach: Direct flights from Islamabad, or a scenic road trip via the "
            "Karakoram Highway. Highlights include Shangrila Resort, Deosai Plains, "
            "Shigar Fort, and Upper Kachura Lake."
        ),
    },
    {
        "slug": "swat-valley",
        "title": "Swat Valley",
        "text": (
            "Swat Valley combines alpine scenery with rich Buddhist and Islamic "
            "history, known as the 'Switzerland of Pakistan' for its lush green "
            "landscapes. Rivers, meadows, and pine forests make it a favorite for "
            "both nature lovers and history enthusiasts exploring its ancient stupas "
            "and monasteries. Best time to visit: April to September. How to reach: "
            "A 4-5 hour drive from Islamabad via the Swat Expressway. Highlights "
            "include Malam Jabba, Kalam Valley, Mingora Bazaar, and Ushu Forest."
        ),
    },
]


def get_embedding(text: str):
    return embed_model.encode(text).tolist()


def main():
    if qdrant_client.collection_exists(COLLECTION):
        qdrant_client.delete_collection(COLLECTION)

    qdrant_client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    points = []
    for i, dest in enumerate(destinations):
        vector = get_embedding(dest["text"])
        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={
                    "slug": dest["slug"],
                    "title": dest["title"],
                    "text": dest["text"],
                },
            )
        )
        print(f"Embedded: {dest['title']}")

    qdrant_client.upsert(collection_name=COLLECTION, points=points)
    print(f"\nDone. Uploaded {len(points)} destinations to Qdrant.")


if __name__ == "__main__":
    main()