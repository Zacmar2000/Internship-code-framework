tools = [
        {
        "name": "retrieve_context",
        "type": "function",
        "description": "Recupera parti di documenti (chunk di testo) semanticamente rilevanti rispetto alla query da una knowledge base indicizzata. Usare questo tool quando servono informazioni specifiche provenienti dai documenti del sistema.",
        "parameters": {
            "type": "object",
            "properties": {
            "query": {
                "type": "string",
                "description": "Domanda o argomento da cercare nella knowledge base. Deve essere una query descrittiva che rappresenti le informazioni da recuperare."
            },
            "year": {
                "type": "integer",
                "minimum": 2024,
                "maximum": 2026,
                "description": "Filtro per anno dei documenti. La ricerca verrà limitata ai documenti validi in quell'anno."
                }
            },
            "required": ["query", "year"],
            "additionalProperties": False
        },
        "strict": True
        },
        {
        "name": "retrieve_normativa",
        "type": "function",
        "description": "Recupera il testo di una norma specifica filtrando per tipologia, numero, anno e riferimenti interni (articolo o comma). Restituisce sempre la versione più recente valida per l'anno richiesto.",
        "parameters": {
            "type": "object",
            "properties": {
                "tipologia": {
                    "type": "string",
                    "description": "Tipo di atto normativo (es. Legge, Decreto Legge, Decreto Legislativo, Regio Decreto)"
                },
                "numero": {
                    "type": "integer",
                    "description": "Numero dell'atto normativo"
                },
                "anno_legge": {
                    "type": "integer",
                    "description": "Anno di emanazione della norma"
                },
                # "data": {
                #     "type": ["string", "null"],
                #     "description": "Data completa della norma nel formato GG/MM/AAAA (opzionale)"
                # },
                "reference": {
                    "type": "object",
                    "description": "Riferimento interno: specificare almeno articolo o comma.",
                    "properties": {
                        "articolo": {
                            "type": "integer"
                        },
                        "comma": {
                            "type": "integer"
                        }
                    },
                    "minProperties": 1,
                    "additionalProperties": False
                },
                # "suffix": {
                #     "type": ["string", "null"],
                #     "description": "Eventuale suffisso del comma (es. bis, ter)"
                # },
                "year": {
                    "type": "integer",
                    "description": "Anno di validità"
                }
            },
            "required": ["tipologia", "anno_legge", "reference", "year"]
        }
    },
    {
    "name": "get_day_of_week",
    "type": "function",
    "description": "Restituisce il giorno della settimana corrispondente alla data fornita. Usare questo tool quando serve conoscere il giorno della settimana di una data specifica.",
    "parameters": {
        "type": "object",
        "properties": {
        "date": {
            "type": "string",
            "description": "Data in formato GG/MM/AAAA (es. 17/02/2026).",
            "pattern": "^\\d{2}/\\d{2}/\\d{4}$"
        }
        },
        "required": ["date"],
        "additionalProperties": False
    },
    "strict": True
    },
    {
    "name": "get_vocabulary",
    "type": "function",
    "description": "Recupera definizioni ufficiali dal vocabolario del comune per chiarire termini della query",
    "parameters": {
        "type": "object",
        "properties": {
        "query": {
            "type": "string",
            "description": "Termine da cercare nel vocabolario."
        }
        },
        "required": ["query"],
        "additionalProperties": False
    },
    "strict": True
    },
    ]