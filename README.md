# Multilingual Semantic Search — E5 (Small & Large) + Qdrant

A semantic search system for routing banking customer-service queries, using multilingual E5 embeddings and Qdrant as the vector store. Built and tested on English, Hindi, and Hinglish queries.

## TL;DR

- Embeds queries and a labeled query dataset with **E5 multilingual (small and large)**, then does semantic search in **Qdrant**.
- For each query, returns the **top 3 matching chunks** plus a predicted label and confidence score — the top chunks matter more than a single "final answer" because they're meant to be handed to an LLM downstream, which makes the final routing decision, not raw semantic search alone.
- **E5-small outperformed E5-large** on this test set: small scored **6/7 (85.7%)** accuracy, large scored **5/7 (71.4%)**.

## Use case

Banking-adjacent customer service automation: given a user query in any supported language, find the most similar banking intents from a labeled dataset to route the query or trigger the right response. The pipeline is dataset-agnostic — swap in a different labeled dataset and the use case changes with it.

## Results

| Model | Dimensions | Accuracy |
|---|---:|---:|
| E5-multilingual-**small** | 384 | 6/7 (85.7%) |
| E5-multilingual-**large** | 1024 | 5/7 (71.4%) |

### Per-query results

| Query | Language | Actual Label | E5-Small Predicted | E5-Large Predicted |
|---|---|:---:|:---:|:---:|
| "I lost my debit card" | EN | 41 | ✅ 41 | ✅ 41 |
| "I want to change my card pin" | EN | 21 | ✅ 21 | ✅ 21 |
| "How much cash can I deposit in one go?" | EN | 58 | ✅ 58 | ✅ 58 |
| "Where is the nearest branch?" | EN | 3 | ✅ 3 | ✅ 3 |
| "My card payment was declined" | EN | — (unlabeled) | 25 | 25 |
| "I want to know my interest rate" | EN | 32 | ✅ 32 | ❌ 70 |
| "ATM se paisa nahi nikla" | Hinglish | 20 | ❌ 26 | ❌ 26 |
| "मैंने अपना कार्ड खो दिया है" | Hindi | 41 | ✅ 41 | ✅ 41 |

Both models struggled on the same Hindi/Hinglish ATM query, and E5-large additionally missed the interest-rate query (matching it to a "source of funds" intent instead). Scores across the board sat in a fairly narrow band (0.82–0.97 top similarity), which is part of why the workflow logic below leans toward asking for clarification rather than auto-executing.

## Confidence & workflow decision logic

Each query gets a **similarity** score (raw top cosine similarity) and a **confidence** score (derived from the gap between the top match and the runner-up), which together decide what the system should do next:

```python
def decide_workflow(similarity, confidence):
    if similarity >= 0.85 and confidence >= 0.70:
        return "AUTO_EXECUTE_INTENT"      # clear match
    elif similarity >= 0.80 and confidence < 0.50:
        return "ASK_CLARIFICATION"        # multiple good, similar options
    elif similarity < 0.60:
        return "FALLBACK_TO_HUMAN"        # poor match regardless of confidence
    else:
        return "ASK_CLARIFICATION"        # medium zone
```

In this test run, confidence scores consistently landed in the 0.23–0.24 range, so nearly every query was routed to `ASK_CLARIFICATION` — meaning the top match was usually good, but close enough to alternatives that the system preferred a clarifying question over auto-executing.

## Tech stack

| Component | Choice |
|---|---|
| Embedding models | `intfloat/multilingual-e5-large` (1024-d) and `intfloat/multilingual-e5-small` (384-d) |
| Vector database | [Qdrant](https://qdrant.tech/) |
| Data format | Parquet (HuggingFace dataset) |
| Framework / libs | `sentence-transformers`, `qdrant-client`, `pandas`, `pyarrow`, `FastAPI` |

## Dataset — Banking77

- **Domain:** Banking customer-service queries
- **Size:** 5,003 train / 3,080 test samples
- **Classes:** 77 intent categories (e.g. `change_pin`, `balance`, `atm_location`)
- **Structure:** two columns — `text` (query) and `label` (intent class, 0–76)
- Swappable — the pipeline isn't tied to banking; changing the dataset changes the use case.

Note: Banking77 itself only contains English queries. Multilingual coverage in this pipeline currently comes entirely from the embedding model, not from the training data — see [Limitations](#limitations--next-steps).

## How it works

1. **Data loading** — read the Banking77 parquet files into pandas DataFrames.
2. **Embedding generation** — prefix every text with `"query: "` (E5's required format for its asymmetric search design), batch-encode with the E5 model, and pull out normalized numpy vectors.
3. **Vector DB setup** — connect to Qdrant, delete any existing collection with the same name, and recreate it with the right vector size (384 for small, 1024 for large) and cosine distance.
4. **Indexing** — build a `PointStruct` per row (id, vector, payload with `text` + `label`) and batch-upload to Qdrant.
5. **Semantic search** — encode the incoming query (same `"query: "` prefix), search Qdrant for the top-k nearest vectors, and return them with scores and metadata.

```python
def semantic_search(query, top_k=3, label_filter=None):
    query_vector = embedder.encode(query, is_query=True)[0]
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=top_k,
        with_payload=True
    )
    return results
```

## Setup

```bash
pip install -U sentence-transformers qdrant-client pandas pyarrow fastapi
pip install "httpx<0.28" "pandas<3.0" --upgrade
```

Qdrant runs in server mode (`localhost:6333`, e.g. via Docker) in this implementation, so a Qdrant instance needs to be running before the notebook connects to it. An in-memory (`:memory:`) mode also works for quick local testing/demos, but data won't persist between runs.

## Implementation notes

- **E5** = "EmbEddings from bidirEctional Encoder rEpresentations."
- Multilingual coverage spans 100+ languages, including Hindi, Tamil, Telugu, Marathi, Gujarati, Bengali, Punjabi, Kannada, and Malayalam.
- E5 uses an **asymmetric search design**: `"query: "` for queries and `"passage: "` for documents. This implementation currently uses `"query: "` for both — worth revisiting, since using the correct prefix for indexed passages may improve retrieval quality.
- Collection recreation (`delete_collection()` → `create_collection()`) keeps the setup idempotent, so the same code runs cleanly against any dataset or model swap.

## Limitations & next steps

- English and Hinglish queries performed well; some Hindi queries did not return the expected results.
- Likely cause: Banking77 has no queries in vernacular languages, so the *only* multilingual-aware part of the pipeline is the embedder itself — the indexed data it's matching against is all English.
- Next step: test against a dataset that includes genuine vernacular-language queries, not just a multilingual embedder over English data.

## References

- E5 paper: *Text Embeddings by Weakly-Supervised Contrastive Pre-training*
- [Qdrant documentation](https://qdrant.tech/documentation/)
- [Banking77 dataset](https://huggingface.co/datasets/PolyAI/banking77) (HuggingFace)
- [Sentence Transformers](https://www.sbert.net/)

## Author

Ona Dubey
