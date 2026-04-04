import os
import chromadb


def get_chroma_client(path="models/chroma"):
    os.makedirs(path, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def get_collection(name="cris_embeddings", path="models/chroma"):
    client = get_chroma_client(path)
    return client.get_or_create_collection(name)


def upsert_embeddings(embeddings, metadatas, ids, collection_name="cris_embeddings", path="models/chroma"):
    col = get_collection(collection_name, path)
    col.upsert(embeddings=embeddings, metadatas=metadatas, ids=ids)
    return col


def query_embeddings(query_embedding, top_n=3, collection_name="cris_embeddings", path="models/chroma"):
    col = get_collection(collection_name, path)
    res = col.query(query_embeddings=[query_embedding], n_results=top_n)
    return res

