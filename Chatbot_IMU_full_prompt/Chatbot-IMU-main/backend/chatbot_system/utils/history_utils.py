import json
from chatbot_system.config import redis_client

async def get_history(session_id: str):
    data = await redis_client.get(session_id)  # <- await qui
    return json.loads(data) if data else []

# Funzione asincrona per salvare la cronologia
async def save_history(session_id: str, history: list, ttl: int = 3600):
    history = history[-20:]  # mantieni solo gli ultimi 20 messaggi
    await redis_client.set(session_id, json.dumps(history), ex=ttl) 