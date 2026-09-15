import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import fakeredis.aioredis as fakeredis

load_dotenv()

# -------------------
# API
# -------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------
# Modelli
# -------------------

reasoning = True
reasoning_effort = "low"

if reasoning:
    model = "gpt-5-mini"
else:
    model = "gpt-5.1-chat-latest"

# -------------------
# logging / audit
# -------------------

audit = True
logs = True


# -------------------
# Prompts
# -------------------

PROMPT_PATH = Path("Prompts")

main_prompt = (PROMPT_PATH / "main_prompt.txt").read_text(encoding="utf-8")

prompt_aliquote = (PROMPT_PATH / "Prompts_fonti" / "Aliquote.txt").read_text(encoding="utf-8")
prompt_nazionale = (PROMPT_PATH / "Prompts_fonti" / "Regolamento_Nazionale.txt").read_text(encoding="utf-8")
prompt_comunale = (PROMPT_PATH / "Prompts_fonti" / "Regolamento_Comunale.txt").read_text(encoding="utf-8")
prompt_file_rimanenti = (PROMPT_PATH / "Prompts_fonti" / "File_rimanenti.txt").read_text(encoding="utf-8")
prompt_vocab = (PROMPT_PATH / "Prompts_fonti" / "Vocabolario_reduced.txt").read_text(encoding="utf-8")

# -------------------
# History (Redis)
# -------------------

# redis_client "finto" in memoria
redis_client = fakeredis.FakeRedis(decode_responses=True)
history_ttl = 3600  # 1 ora