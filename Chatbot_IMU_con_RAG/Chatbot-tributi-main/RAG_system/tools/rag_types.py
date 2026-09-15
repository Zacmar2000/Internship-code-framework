from typing import TypedDict, List, Optional
from typing_extensions import NotRequired
from dataclasses import dataclass

class ChunkMetadata(TypedDict):
    document: str
    title: str
    page: int
    sub_chunk: int
    valenza: str
    score: NotRequired[float]
    categoria: NotRequired[str]
    periodo: NotRequired[List[str]]
    anno: NotRequired[List[int]]

class Chunk(TypedDict):
    embedding_id: int
    text: str
    metadata: ChunkMetadata
    source_pdf: str

@dataclass
class QueryWithYear:
    query: str
    year: Optional[int]

Chunks = List[Chunk]