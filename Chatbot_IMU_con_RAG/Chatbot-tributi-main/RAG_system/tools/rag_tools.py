from typing import Optional
import re
import asyncio
import time
from datetime import datetime
from RAG_system.retrieval.chunk_reconstruction import extract_hierarchy, dict_title_to_full_text
from RAG_system.retrieval.compress_extraction import compress_one_category


class RAGTools:

    def __init__(self, retrieval_engine, model, reasoning, reasoning_effort):
        self.retrieval_engine = retrieval_engine
        self.model = model
        self.reasoning = reasoning
        self.reasoning_effort = reasoning_effort
        self.retrieved_docs_past = {}

    def retrieve_context(self, query: str, year: int, k: int = 3):


        docs_results = self.retrieval_engine.search(
            query=query,
            year=year,
            k=k
        )

        category_results = {}

        agent_sources = []

        for i,docs in enumerate(docs_results):

            sources = []

            for doc in docs:

                metadata = doc["metadata"]
                score = metadata["score"]

                sources.append({
                    "text": doc["text"],
                    "embedding_id": doc["embedding_id"],
                    "metadata": metadata,
                    "similarity": score
                })

            for doc in sources:
                agent_sources.append({
                    "emb_id": doc.get("embedding_id"),
                    "similarity": doc.get("similarity", []),
                })
            
            cat = "Regolamento Nazionale (Legge n. 160/2019)" if i==0 else "Resto"

            category_results[cat] = {"sources": sources,
                "year":year,
                "query":query}

        # =====================
        # PREPARAZIONE TASK ASYNC
        # =====================

        tasks_input = []

        for cat, sources in category_results.items():
            
            chunks = sources.get("sources")

            simplified_chunks = [
                {
                    "text": c["text"],
                    "metadata": c.get("metadata", {})
                }
                for c in chunks
            ]

            tasks_input.append({
                "sources": simplified_chunks,
                "user_query": query,
                "category": cat
            })

        # =====================
        # ASYNC COMPRESSION
        # =====================

        compressed_results = asyncio.run(
            compress_all_categories(
                tasks_input,
                self.retrieval_engine.client,
                self.model,
                self.reasoning,
                self.reasoning_effort,
            )
        )

        return {"sources": compressed_results}, {"agent_sources": agent_sources}

    def retrieve_normativa(
        self,
        tipologia: str,
        anno_legge: int,
        year: int,
        numero: Optional[int] = None,
        data: Optional[str] = None,
        articolo: Optional[int] = None,
        comma: Optional[int] = None,
        suffix: Optional[str] = None,
    ):
        def validate_normativa_args(args):
            errors = []

            ALLOWED_NUMERI = [160, 330, 504, 222, 262]

            ALLOWED_ANNI = [2019, 1994, 1992, 1985, 1942]

            #ALLOWED_DATE = ["27/12/2019", "31/5/1994", "30/12/1992", "20/5/1985", "16/03/1942"]

            allowed_tipologie = {
                "legge": "Legge",
                "decreto legge": "Decreto Legge",
                "dl": "Decreto Legge",
                "d.l.": "Decreto Legge",
                "decreto legislativo": "Decreto Legislativo",
                "d.lgs": "Decreto Legislativo",
                "d.lgs.": "Decreto Legislativo",
                "regio decreto": "Regio Decreto"
            }

            # --- TIPOLGIA ---
            tipologia_input = (args.get("tipologia") or "").lower().strip()

            if tipologia_input not in allowed_tipologie:
                errors.append({
                    "field": "tipologia",
                    "message": f"Valore non valido: '{args.get('tipologia')}'",
                    "allowed": list(set(allowed_tipologie.values()))
                })
            else:
                args["tipologia"] = allowed_tipologie[tipologia_input]

            # --- NUMERO ---
            if numero is not None and numero not in ALLOWED_NUMERI:
                errors.append({
                    "field": "numero",
                    "message": f"Numero non valido: {numero}",
                    "allowed_values": ALLOWED_NUMERI
                })

            # --- ANNO LEGGE---
            if anno_legge not in ALLOWED_ANNI:
                errors.append({
                    "field": "anno_legge",
                    "message": f"Anno non valido: {anno_legge}",
                    "allowed_values": ALLOWED_ANNI
                })

            # --- ARTICOLO / COMMA ---
            if args.get("articolo") is None and args.get("comma") is None:
                errors.append({
                    "field": "reference",
                    "message": "Devi specificare almeno articolo o comma",
                    "example": {"articolo": 7} 
                })
            
            # --- YEAR (vincolo minimo) ---
            if year < 2024:
                errors.append({
                    "field": "year",
                    "message": "Year deve essere >= 2024"
                })

            return args, errors
        
        args = {
            "tipologia": tipologia,
            "numero": numero,
            "anno_legge": anno_legge,
            "articolo": articolo,
            "comma": comma,
            "suffix": suffix,
            "year": year
        }

        args, errors = validate_normativa_args(args)

        if errors:
            return {
                "status": "error",
                "message": "Parametri non validi",
                "errors": errors,
                "example_valid_call": {
                    "tipologia": "Decreto Legislativo",
                    "numero": 504,
                    "anno_legge": 1992,
                    "articolo": 7,
                    "year": 2024
                }
            },[]

        all_metadata_filtered = [chunk for chunk in self.retrieval_engine.metadata if chunk.get("metadata",[]).get("categoria") == "Normativa"]

        def parse_date_safe(date_str: str):
            """Parsa una data in formato GG/MM/AAAA anche se ci sono spazi o zeri mancanti"""
            try:
                return datetime.strptime(date_str, "%d/%m/%Y").date()
            except ValueError:
                return datetime.strptime(date_str.strip(), "%d/%m/%Y").date()

        # Se passa la data completa, la parsifichiamo
        use_whole_date = data is not None
        if use_whole_date:
            target_date = parse_date_safe(data)

        

        def is_match(chunk):

            metadata = chunk.get("metadata")
            if not metadata:
                return False

            # Tipologia
            if metadata.get("tipologia", "").lower() != tipologia.lower():
                return False

            # Data
            data_str = metadata.get("data")
            if not data_str:
                return False

            chunk_date = parse_date_safe(data_str)

            if use_whole_date:
                if chunk_date != target_date: # pyright: ignore[reportPossiblyUnboundVariable]
                    return False
            else:
                if chunk_date.year != anno_legge:
                    return False

            # Anno validità
            anni_validita = metadata.get("anno")
            if anni_validita and year not in anni_validita:
                return False

            # Numero
            if numero is not None:
                if str(metadata.get("numero")) != str(numero):
                    return False

            # Comma
            if comma is not None:
                title = metadata.get("title", "").lower()

                comma_text = f"comma {comma}".lower()
                if suffix:
                    comma_text += f"-{suffix}"

                if title != comma_text:
                    return False
                if articolo is not None:
                    title_1 = metadata.get("title_1", "")

                    pattern = rf"^(art\.?|articolo)\s*{articolo}\b"

                    if not re.match(pattern, title_1, re.IGNORECASE):
                        return False
            
            # Articolo
            elif articolo is not None:
                title = metadata.get("title", "")
                title_1 = metadata.get("title_1", "")

                pattern = rf"^(art\.?|articolo)\s*{articolo}\b"

                if not (re.match(pattern, title, re.IGNORECASE) or re.match(pattern, title_1, re.IGNORECASE)):
                    return False
            
            else:
                return False


            return True
        
        filtered_norm = [chunk for chunk in all_metadata_filtered if is_match(chunk)]

        if filtered_norm:

            metadata = filtered_norm[0].get("metadata",[])

            document_name = metadata["document"]
            valenza = metadata['valenza']

            hierarchy = extract_hierarchy(metadata)

            if len(filtered_norm)>1:
                filtered_norm.sort(
                        key=lambda s: (s['metadata']['sub_chunk'])
                    )
                filtered_norm_dict = dict_title_to_full_text(filtered_norm)
                if len(filtered_norm_dict) > 1:

                    def parse_end_period(key):
                        # key = (title, periodo)
                        periodo = key[1]

                        if isinstance(periodo, list):
                            end_date = periodo[1]
                        else:
                            end_date = periodo[1]

                        return datetime.strptime(end_date, "%d/%m/%Y").date()

                    # prende la chiave con periodo più avanti
                    latest_key = max(filtered_norm_dict.keys(), key=parse_end_period)
                    periodo = latest_key[1]

                    text_final = filtered_norm_dict[latest_key].get("text")
                    embedding_ids = filtered_norm_dict[latest_key].get("embedding_ids")
                else:
                    
                    final_values = next(iter(filtered_norm_dict.values()))
                    text_final = final_values.get("text")
                    embedding_ids = final_values.get("embedding_ids")

                    periodo = next(iter(filtered_norm_dict.keys()))[1]

            else:
                text_final = filtered_norm[0].get('text')

                periodo = metadata.get("periodo")

                embedding_ids = [filtered_norm[0].get('embedding_id')]
            



            return {
                    "documento": document_name,
                    "paragrafi": hierarchy,
                    "valenza": valenza,
                    "periodo": periodo,
                    "testo": text_final,
                }, embedding_ids
        else:
            return {},[]
        

    def retrieve_vocabulary(self,arguments):

        results_voc = ""

        check_presence = []

        for arg in arguments:
            if not (arg in self.retrieval_engine.titles_vocabulary):
                check_presence.append(0)
                continue
            
            else:
                choosen = [vocab for vocab in self.retrieval_engine.vocabolario['IMU'] if vocab['title'] == arg][0]
                results_voc += f"\n\n### {choosen['title']}\n{choosen['text']}"
                check_presence.append(1)
        
        return { "Vocabolario": results_voc}, check_presence
        


async def timed_compress(task, client, model, reasoning, reasoning_effort, session= None):
    start = time.perf_counter()
    
    result = await compress_one_category(**task,
                                         client=client,
                                         model=model,
                                         reasoning=reasoning,
                                         reasoning_effort=reasoning_effort,
                                         session=session)
    
    end = time.perf_counter()
    print(f"{task.get('category', 'unknown')} -> {end - start:.2f}s")
    
    return result

async def compress_all_categories(tasks_input, client, model, reasoning,reasoning_effort, session = None):
    tasks = [
        timed_compress(task, client, model, reasoning, reasoning_effort, session)
        for task in tasks_input
    ]

    results = await asyncio.gather(*tasks)

    return {
        cat: text
        for cat, text in results # type: ignore
        if text
    }