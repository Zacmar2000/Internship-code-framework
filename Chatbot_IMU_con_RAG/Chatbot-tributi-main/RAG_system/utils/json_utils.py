import json

def safe_json_load(text: str):
    decoder = json.JSONDecoder()
    text = text.strip()

    for i in range(len(text)):
        try:
            obj, idx = decoder.raw_decode(text[i:])
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            continue

    raise ValueError("Nessun JSON valido trovato.")