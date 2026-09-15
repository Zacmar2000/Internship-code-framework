from RAG_system.retrieval.retrieval_engine import RetrievalEngine

from RAG_system.retrieval.compress_extraction import compress_one_category
from RAG_system.retrieval.reranker import regroup_by_category, global_rerank

from pathlib import Path
import json
import asyncio
import time

cat_dict={'Aliquote': 'Aliquote comunali', 'Istruzioni': 'Istruzioni compilazione', 'Normativa': "Regolamento nazionale e altre Normative",
          'Privacy': 'Privacy', 'Regolamento': 'Regolamento comunale'}

async def timed_compress(task, client, model, reasoning, reasoning_effort, session):
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

async def compress_all_categories(tasks_input, client, model, reasoning,reasoning_effort, session):
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

def data_retrieval_new(user_query:str,
                       session,
                       retrieval_engine: RetrievalEngine,
                       reasoning,
                       reasoning_effort,
                       model,
                       reranker,
                       k= 3,
                       audit_trace=None,
                       logger=None,
                       log_tool_result=None,
                       ):

    prompt_dir = Path(__file__).resolve().parents[2] / "Prompts"
    prompt = (prompt_dir / "prompt_decision_query.txt").read_text()

    client = retrieval_engine.client

    input_messages = [
        {"role": "system", "content": prompt}
    ]

    input_messages.extend(session.get_history()[-6:])
    input_messages.append({"role": "user", "content": user_query})

    client = retrieval_engine.client

    params = {
        "model": model,
        "input": input_messages,
    }

    if reasoning:
        params["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**params)

    try:
        router_output = json.loads(response.output_text)
    except:
        router_output = {
            "Aliquote": {"use": False, "query": None, "year": None},
            "Istruzioni": {"use": False, "query": None, "year": None},
            "Privacy": {"use": False, "query": None, "year": None},
            "Normativa": {"use": True, "query": user_query, "year": None},
            "Regolamento": {"use": True, "query": user_query, "year": None},
        }

    category_results = {}

    for cat, data in router_output.items():

        if not data["use"]:
            continue

        search_query = data["query"]
        year = data["year"]

        # Log debug
        if logger:
            logger.debug(f"Retrieval iniziale per {cat}:")
            logger.debug(f"Args: {(year,search_query)}")

        docs = retrieval_engine.search_new(
            query=search_query,
            category=cat,
            k=2*k,
            year=year
        )

        sources = []
        for doc in docs:
            metadata = doc["metadata"]

            sources.append({
                "text": doc["text"],
                "embedding_id": doc["embedding_id"],
                "metadata": metadata,
            })

        # Log debug
        if logger:
            if sources:
                logger.debug(f"Estratti {len(sources)} testi")
            else:
                logger.debug("Nessun testo rilevante")

        category_results[cat] = {"sources": sources,
                                 "year":year,
                                 "query":search_query}


    # =====================
    # GLOBAL RERANK
    # =====================

    top_chunks = global_rerank(
        user_query=user_query,
        category_results=category_results,
        reranker=reranker,
        top_k=k,
        session=session
    )

    if logger:
        if top_chunks:
            logger.debug("Reranking globale eseguito!")
        else:
            logger.debug("Nessun testo dopo reranking")

    grouped_chunks = regroup_by_category(top_chunks)

    # =====================
    # PREPARAZIONE TASK ASYNC
    # =====================

    tasks_input = []

    for cat, chunks in grouped_chunks.items():

        simplified_chunks = [
            {
                "text": c["text"],
                "metadata": c.get("metadata", {})
            }
            for c in chunks
        ]

        tasks_input.append({
            "sources": simplified_chunks,
            "user_query": user_query,
            "category": cat
        })

        agent_sources = []

        for doc in chunks:
            agent_sources.append({
                "emb_id": doc.get("embedding_id"),
                "similarity": doc.get("similarity"),
            })


        if audit_trace and chunks:
            audit_trace.add_starting_call(
                category=cat,
                args={"year":chunks[0].get("year"),"query":chunks[0].get("query")},
                result=agent_sources
            )

    # =====================
    # ASYNC COMPRESSION
    # =====================

    compressed_results = asyncio.run(
        compress_all_categories(
            tasks_input,
            retrieval_engine.client,
            model,
            reasoning,
            reasoning_effort,
            session=session
        )
    )

    if logger:
        if compressed_results:
            logger.debug("Estrazione testo eseguita!")
        else:
            logger.debug("Nessun testo dopo estrazione")

    # =====================
    # MERGE FINALE
    # =====================

    results = ""

    for cat, text in compressed_results.items():
        results += f"\n\n### {cat_dict[cat]}\n{text}"

        if audit_trace:
            audit_trace.add_starting_text(
                category=cat,
                result=text
            )

    return results