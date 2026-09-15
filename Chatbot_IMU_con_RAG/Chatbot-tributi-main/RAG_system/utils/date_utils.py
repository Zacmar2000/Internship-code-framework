from datetime import datetime

mesi = [
    "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
    "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"
]


def get_current_datetime():
    now = datetime.now()
    return f"{now.day:02d} {mesi[now.month - 1]} {now.year}"

def get_day_of_week(date: str) -> str:
    if not isinstance(date, str):
        return "Errore: la data deve essere una stringa nel formato GG/MM/AAAA."

    try:
        data = datetime.strptime(date.strip(), "%d/%m/%Y")
    except ValueError:
        return "Errore: formato non valido. Usa GG/MM/AAAA (es. 17/02/2026)."

    giorni_it = [
        "Lunedì",
        "Martedì",
        "Mercoledì",
        "Giovedì",
        "Venerdì",
        "Sabato",
        "Domenica"
    ]

    return giorni_it[data.weekday()]