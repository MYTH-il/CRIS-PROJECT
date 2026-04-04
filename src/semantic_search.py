import os
import json
import time
from datetime import datetime
import numpy as np
import requests
from src.vector_store import upsert_embeddings, query_embeddings


class SemanticSearchEngine:
    def __init__(self, model_name=None, token=None):
        self.model = model_name or os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.token = token or os.getenv("HF_API_TOKEN")
        self.embeddings = None
        self.records = None

    def _hf_embed_batch(self, texts):
        if not self.token:
            return None, "HF_API_TOKEN not set."
        try:
            resp = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"inputs": texts},
                timeout=30,
            )
            if resp.status_code != 200:
                return None, f"HF API error: {resp.status_code}"
            data = resp.json()
        except Exception as e:
            return None, f"HF request failed: {str(e)}"

        if not isinstance(data, list):
            return None, "Unexpected HF response."

        # Data can be list[seq_len][dim] or list[batch][seq_len][dim]
        if data and isinstance(data[0], list) and data and data and isinstance(data[0][0], list):
            # batch
            embeds = []
            for seq in data:
                arr = np.array(seq, dtype=np.float32)
                embeds.append(arr.mean(axis=0))
            return np.vstack(embeds), None
        if data and isinstance(data[0], list):
            arr = np.array(data, dtype=np.float32)
            return arr.mean(axis=0).reshape(1, -1), None
        return None, "Unexpected HF response shape."

    def build_index(self, df, text_col="incident_description", sample_size=2000, batch_size=8):
        if text_col not in df.columns:
            return False, f"Missing column: {text_col}"

        subset = df.dropna(subset=[text_col]).copy()
        if len(subset) == 0:
            return False, "No valid text rows."

        if sample_size and len(subset) > sample_size:
            subset = subset.sample(sample_size, random_state=42)

        texts = subset[text_col].tolist()
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            emb, err = self._hf_embed_batch(batch)
            if err:
                return False, err
            embeddings.append(emb)
            time.sleep(0.2)

        self.embeddings = np.vstack(embeddings)
        self.records = subset.reset_index(drop=True)
        return True, f"Indexed {len(self.records)} records."

    def build_chroma_index(self, df, text_col="incident_description", sample_size=2000, batch_size=8):
        if text_col not in df.columns:
            return False, f"Missing column: {text_col}"
        subset = df.dropna(subset=[text_col]).copy()
        if len(subset) == 0:
            return False, "No valid text rows."
        if sample_size and len(subset) > sample_size:
            subset = subset.sample(sample_size, random_state=42)

        texts = subset[text_col].tolist()
        ids = [f"case_{i}" for i in range(len(texts))]
        metadatas = []

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            emb, err = self._hf_embed_batch(batch)
            if err:
                return False, err
            embeddings.append(emb)
            time.sleep(0.2)

        embeddings = np.vstack(embeddings).tolist()
        for _, row in subset.iterrows():
            metadatas.append({
                "crime": row.get("crime_type", "Unknown"),
                "location": row.get("incident_location", "Unknown"),
                "date": row.get("report_date", "Unknown"),
                "snippet": str(row.get(text_col, ""))[:150] + "...",
            })

        upsert_embeddings(embeddings=embeddings, metadatas=metadatas, ids=ids)
        return True, f"Chroma index built with {len(ids)} records."

    def save_index(self, folder_path="models/semantic_index"):
        if self.embeddings is None or self.records is None:
            return False, "Nothing to save."
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, "embeddings.npy"), self.embeddings)
        records_path = os.path.join(folder_path, "records.jsonl")
        with open(records_path, "w", encoding="utf-8") as f:
            for _, row in self.records.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        meta = {
            "model": self.model,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "count": int(self.embeddings.shape[0]),
        }
        with open(os.path.join(folder_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return True, "Index saved."

    def load_index(self, folder_path="models/semantic_index"):
        emb_path = os.path.join(folder_path, "embeddings.npy")
        records_path = os.path.join(folder_path, "records.jsonl")
        if not os.path.exists(emb_path) or not os.path.exists(records_path):
            return False, "Index files not found."
        self.embeddings = np.load(emb_path)
        records = []
        with open(records_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        self.records = np.array(records, dtype=object)
        if len(records) > 0:
            import pandas as pd
            self.records = pd.DataFrame(records)
        return True, "Index loaded."

    def query(self, text, top_n=3):
        if self.embeddings is None or self.records is None:
            return None, "Index not built."
        emb, err = self._hf_embed_batch([text])
        if err:
            return None, err
        query_vec = emb[0]
        norms = np.linalg.norm(self.embeddings, axis=1) * (np.linalg.norm(query_vec) + 1e-9)
        sims = np.dot(self.embeddings, query_vec) / (norms + 1e-9)
        top_idx = np.argsort(sims)[-top_n:][::-1]
        results = []
        for idx in top_idx:
            row = self.records.iloc[idx]
            results.append({
                "similarity": float(sims[idx] * 100),
                "crime": row.get("crime_type", "Unknown"),
                "location": row.get("incident_location", "Unknown"),
                "date": row.get("report_date", "Unknown"),
                "snippet": str(row.get("incident_description", ""))[:150] + "...",
            })
        return results, None

    def query_chroma(self, text, top_n=3):
        emb, err = self._hf_embed_batch([text])
        if err:
            return None, err
        query_vec = emb[0].tolist()
        res = query_embeddings(query_embedding=query_vec, top_n=top_n)
        results = []
        if res and res.get("metadatas"):
            for md in res["metadatas"][0]:
                results.append({
                    "similarity": 0.0,
                    "crime": md.get("crime", "Unknown"),
                    "location": md.get("location", "Unknown"),
                    "date": md.get("date", "Unknown"),
                    "snippet": md.get("snippet", ""),
                })
        return results, None
