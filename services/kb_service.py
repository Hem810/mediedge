"""Knowledge base service - BM25 retrieval over WHO IMCI protocols + drug formulary."""
from __future__ import annotations
import json
import math
import re

from config import settings


class KnowledgeBase:
    def __init__(self) -> None:
        self._entries: list[dict] = []
        self._drugs: list[dict] = []
        self._referral_centres: list[dict] = []
        self._loaded: bool = False

    def load(self) -> None:
        if self._loaded:
            return
        data_dir = settings.DATA_DIR

        imci_path = data_dir / "who_imci.json"
        if imci_path.exists():
            self._entries = json.loads(imci_path.read_text()).get("entries", [])

        drug_path = data_dir / "drug_formulary.json"
        if drug_path.exists():
            self._drugs = json.loads(drug_path.read_text()).get("drugs", [])

        ref_path = data_dir / "referral_centres.json"
        if ref_path.exists():
            self._referral_centres = json.loads(ref_path.read_text()).get("centres", [])

        self._loaded = True

    def retrieve(self, query: str, age_months: int = 60, top_k: int = 5) -> list[dict]:
        self.load()
        terms = self._tokenise(query)

        candidates = [
            e for e in self._entries
            if e.get("age_min_months", 0) <= age_months <= e.get("age_max_months", 1200)
        ] or self._entries

        scores = self._bm25(terms, candidates)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        results = [entry for entry, score in ranked[:top_k] if score > 0]
        return results or candidates[:top_k]

    def _bm25(self, terms, docs, k1=1.5, b=0.75):
        if not docs or not terms:
            return [0.0] * len(docs)
        doc_texts = [
            f"{d.get('title', '')} {d.get('summary', '')} {' '.join(d.get('keywords', []))}"
            for d in docs
        ]
        tokenised = [self._tokenise(t) for t in doc_texts]
        avg_len = sum(len(t) for t in tokenised) / max(len(tokenised), 1)

        scores = []
        for doc_tokens in tokenised:
            doc_len = len(doc_tokens)
            score = 0.0
            for term in terms:
                tf = doc_tokens.count(term)
                if tf == 0:
                    continue
                df = sum(1 for dt in tokenised if term in dt)
                idf = math.log((len(docs) - df + 0.5) / (df + 0.5) + 1)
                tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_len))
                score += idf * tf_norm
            scores.append(score)
        return scores

    def _tokenise(self, text: str) -> list[str]:
        return [w for w in re.sub(r"[^\w\s]", " ", text.lower()).split() if len(w) > 2]

    def lookup_drug(self, name: str) -> dict | None:
        self.load()
        name_l = name.lower()
        for drug in self._drugs:
            if name_l in drug.get("generic_name", "").lower():
                return drug
        return None

    def get_nearest_centres(self, urgency: str = "low", limit: int = 3) -> list[dict]:
        self.load()
        return sorted(
            self._referral_centres,
            key=lambda c: c.get("distance_from_pilani_km", 999),
        )[:limit]

    def build_context(self, query: str, age_months: int) -> str:
        entries = self.retrieve(query, age_months)
        if not entries:
            return "No specific protocol matched."
        return "\n".join([
            f"- [{e.get('subcategory', '')}] {e['title']}: {e['summary']}"
            for e in entries
        ])


kb = KnowledgeBase()
