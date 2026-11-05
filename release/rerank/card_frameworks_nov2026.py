"""Add a 'RAG frameworks' usage section (LangChain, LlamaIndex, Haystack; verified by rerank/integration_check.py, job b96e1231) to the
reranker cards (README only) and to release/rerank/make_rr_card.py."""
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

SEC = """RAG frameworks (checked November 2026 with langchain 1.4, llama-index-core 0.14 and haystack-ai 3.3; all three load the model
through sentence-transformers' `CrossEncoder`):

```python
# LangChain: pip install langchain-community langchain-classic sentence-transformers
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
reranker = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name="{REPO}"), top_n=3)
top_docs = reranker.compress_documents(documents, query)   # use with ContextualCompressionRetriever

# LlamaIndex: pip install llama-index-core sentence-transformers
from llama_index.core.postprocessor import SentenceTransformerRerank
reranker = SentenceTransformerRerank(model="{REPO}", top_n=3)   # node_postprocessors=[reranker]

# Haystack 3: pip install haystack-ai sentence-transformers-haystack
from haystack_integrations.components.rankers.sentence_transformers import SentenceTransformersSimilarityRanker
ranker = SentenceTransformersSimilarityRanker(model="{REPO}", top_k=3)
```

"""
if __name__ == "__main__":
    api = HfApi()
    for size in ("small", "base"):
        repo = f"Horizon-Labs/multilingual-reranker-{size}"; info = api.model_info(repo)
        t = open(hf_hub_download(repo, "README.md", revision=info.sha)).read()
        if "RAG frameworks (checked" in t:
            print(repo, "already"); continue
        assert t.count("## Evaluation") == 1
        print(repo, api.create_commit(repo, [CommitOperationAdd("README.md", t.replace("## Evaluation", SEC.replace("{REPO}", repo) + "## Evaluation").encode())],
                                      parent_commit=info.sha, commit_message="Card: LangChain / LlamaIndex / Haystack usage (verified)").oid)
    p = "release/rerank/make_rr_card.py"; s = open(p).read()
    if "RAG frameworks (checked" not in s:
        s = s.replace("```\n\n## Evaluation", "```\n\n" + SEC + "## Evaluation", 1); open(p, "w").write(s); print("make_rr_card.py updated")
