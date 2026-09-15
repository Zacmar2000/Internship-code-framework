from pydantic import BaseModel

class PromptConfig(BaseModel):
    main_prompt: str
    prompt_nazionale: str
    prompt_comunale: str
    prompt_aliquote: str
    prompt_file_rimanenti: str
    prompt_vocab: str