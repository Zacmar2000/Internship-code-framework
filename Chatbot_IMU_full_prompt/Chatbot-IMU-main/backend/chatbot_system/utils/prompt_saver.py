from pathlib import Path

from chatbot_system.schemas.prompt import PromptConfig

PROMPT_PATH = Path("Prompts")

async def save_modified_prompts(config: PromptConfig):
    (PROMPT_PATH / "main_prompt.txt").write_text(config.main_prompt, encoding="utf-8")
    (PROMPT_PATH / "Prompts_fonti" / "Aliquote.txt").write_text(config.prompt_aliquote, encoding="utf-8")
    (PROMPT_PATH / "Prompts_fonti" / "Regolamento_Nazionale.txt").write_text(config.prompt_nazionale, encoding="utf-8")
    (PROMPT_PATH / "Prompts_fonti" / "Regolamento_Comunale.txt").write_text(config.prompt_comunale, encoding="utf-8")
    (PROMPT_PATH / "Prompts_fonti" / "File_rimanenti.txt").write_text(config.prompt_file_rimanenti, encoding="utf-8")
    (PROMPT_PATH / "Prompts_fonti" / "Vocabolario_reduced.txt").write_text(config.prompt_vocab, encoding="utf-8")