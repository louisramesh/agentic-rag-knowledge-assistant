
# Agentic RAG Knowledge Assistant

**Enterprise‑ready Developer Assistant** powered by:
- **LangGraph** for agent orchestration
- **LangChain** for retrieval utilities
- **Pinecone** vector DB with metadata filtering
- **Azure OpenAI** GPT‑4o / GPT‑4‑turbo and `text‑embedding‑3‑large`
- Example use‑case: internal Developer Knowledge Assistant

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env
python rag_chain_baseline.py          # baseline RAG
python langgraph_agentic_rag.py       # agentic RAG
```

## Repo layout
```
├── langgraph_agentic_rag.py   # full agentic pipeline
├── rag_chain_baseline.py      # simple RAG chain
├── utils/                     # modular helpers
├── notebooks/                 # demos + evaluation
├── data/example_docs/         # sample docs to ingest
└── configs/                   # optional yaml graph config
```

## Next steps to productionize
1. **Metadata modelling** – add `department`, `project_id`, `access_level`, `document_version`…
2. **Secure retrieval** – inject user‑based filter for ABAC in `metadata_filter`
3. **Document pipeline** – robust ingestion + semantic / recursive chunking
4. **Memory** – enable Redis persistence for convo context
5. **Hybrid retrieval** – combine keyword + vector search
6. **Observability** – log retrievals & LLM outputs, add feedback loop
7. **Evaluation** – run `notebooks/RAG_eval.ipynb` to measure precision / recall
8. **Scale & cost** – tune Pinecone index parameters, stream LLM outputs
9. **Compliance** – encrypt PII metadata, add audit logging
10. **CI/CD** – containerise & deploy on AWS ECS/EKS

---

_Generated 2025-06-08T22:39:37.934040Z_
