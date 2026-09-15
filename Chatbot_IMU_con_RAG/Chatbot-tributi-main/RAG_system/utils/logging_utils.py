import json
import logging

def setup_logging(max_tool_result_length: int = 500):
    """
    Configura il logger per il chatbot.

    Args:
        max_tool_result_length: se >0, tronca i risultati dei tool prima di loggarli
    """
    logger = logging.getLogger("rag_agent")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Solo console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    # Evita di aggiungere più handler se già esistono
    if not logger.handlers:
        logger.addHandler(stream_handler)

    # Wrapper helper per troncare tool results
    def log_tool_result(tool_name: str, result, level=logging.DEBUG):
        try:
            text = json.dumps(result, ensure_ascii=False)
        except (TypeError, OverflowError):
            text = str(result)

        if max_tool_result_length and len(text) > max_tool_result_length:
            text = text[:max_tool_result_length] + " ...[truncated]"

        logger.log(level, f"Risultato tool {tool_name}: {text}")

    # Ritorna sia logger che funzione helper
    return logger, log_tool_result