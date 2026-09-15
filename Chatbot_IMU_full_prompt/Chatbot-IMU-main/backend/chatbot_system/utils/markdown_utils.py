import re

def fix_titles(text):
    # Aggiunge una riga vuota dopo ogni titolo markdown (#, ##, ###)
    return re.sub(r'(#+ .+)\n', r'\1\n\n', text)

def fix_paragraphs(text):
    # Trasforma singoli \n in \n\n dove serve
    lines = text.split('\n')
    fixed_lines = []
    for line in lines:
        if line.strip() == "" or line.startswith("#") or line.startswith("- "):
            fixed_lines.append(line)
        else:
            fixed_lines.append(line + "\n")
    return "\n".join(fixed_lines)

def fix_lists(text):
    # Assicura che ogni item di lista abbia una riga vuota dopo
    text = re.sub(r'(\n- .+)', r'\1\n', text)
    return text

def clean_markdown(text):
    text = fix_titles(text)
    text = fix_lists(text)
    text = fix_paragraphs(text)
    text = text.replace("\n", "  \n")  # forza line break
    return text