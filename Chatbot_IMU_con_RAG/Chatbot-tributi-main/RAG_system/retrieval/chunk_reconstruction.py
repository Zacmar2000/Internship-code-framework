import tiktoken
from itertools import groupby


def extract_hierarchy(metadata):
    if not metadata:
        return []

    titles = []
    for key, value in metadata.items():
        if not value:
            continue

        if key == "title" or key.startswith("title_"):
            if key == "title":
                num = 0
            else:
                try:
                    num = int(key.split("_")[1])
                except ValueError:
                    continue

            titles.append((num, value))

    titles.sort(key=lambda x: -x[0])
    return [t[1] for t in titles]


def reconstruct_chunk(sub_chunks, overlap, encoding_name="cl100k_base"):

    enc = tiktoken.get_encoding(encoding_name)

    sub_chunks = sorted(sub_chunks, key=lambda x: x["metadata"]["sub_chunk"])

    reconstructed_tokens = []

    for i, chunk in enumerate(sub_chunks):
        tokens = enc.encode(chunk["text"])

        if i == 0:
            reconstructed_tokens.extend(tokens)
        else:
            reconstructed_tokens.extend(tokens[overlap:])

    return enc.decode(reconstructed_tokens)


def group_by_metadata(chunks, keys):

    def build_key(x):
        metadata = x["metadata"]
        values = []

        for k in keys:
            v = metadata.get(k)

            if isinstance(v, list):
                v = tuple(v)

            values.append(v)

        return tuple(values)

    chunks = sorted(chunks, key=build_key)

    return {
        k: list(g)
        for k, g in groupby(chunks, key=build_key)
    }


def dict_title_to_full_text(final_chunks):

    grouped = group_by_metadata(final_chunks, keys=["title", "periodo"])

    full_chunks = {}

    for section, sub_chunks in grouped.items():

        text = reconstruct_chunk(sub_chunks, overlap=60)

        embedding_ids = [chunk["embedding_id"] for chunk in sub_chunks]

        full_chunks[section] = {
            "text": text,
            "embedding_ids": embedding_ids
        }

    return full_chunks