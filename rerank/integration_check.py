"""Check that the reranker works through the RAG frameworks' standard cross-encoder rerankers (for the model card snippets).
Run in a private venv with langchain(-community/-classic), llama-index-core, haystack-ai, sentence-transformers.
python rerank/integration_check.py MODEL_DIR   -> prints, per framework, the import path that worked and the ranked passages."""
import importlib, json, sys, traceback

M = sys.argv[1]
Q = "How tall is the Eiffel Tower?"
P = ["The Statue of Liberty is 93 metres tall.", "La tour Eiffel a été construite pour l'Exposition universelle de 1889.",
     "The Eiffel Tower is 330 metres tall.", "埃菲尔铁塔高330米。"]
res = {}


def first(paths):
    for mod, name in paths:
        try:
            return getattr(importlib.import_module(mod), name), f"from {mod} import {name}"
        except Exception as e:
            print("import failed:", mod, name, repr(e)[:300], file=sys.stderr)
            continue
    raise ImportError(str(paths))


try:   # LangChain
    HF, p1 = first([("langchain_community.cross_encoders", "HuggingFaceCrossEncoder")])
    CER, p2 = first([("langchain_classic.retrievers.document_compressors", "CrossEncoderReranker"),
                     ("langchain.retrievers.document_compressors", "CrossEncoderReranker"),
                     ("langchain_community.document_compressors", "CrossEncoderReranker")])
    Doc, p3 = first([("langchain_core.documents", "Document")])
    out = CER(model=HF(model_name=M), top_n=2).compress_documents([Doc(page_content=p) for p in P], Q)
    res["langchain"] = dict(imports=[p1, p2, p3], top=[d.page_content for d in out])
except Exception:
    res["langchain"] = dict(error=traceback.format_exc()[-600:])

try:   # LlamaIndex
    STR, p1 = first([("llama_index.core.postprocessor", "SentenceTransformerRerank")])
    from llama_index.core.schema import NodeWithScore, TextNode
    out = STR(model=M, top_n=2).postprocess_nodes([NodeWithScore(node=TextNode(text=p)) for p in P], query_str=Q)
    res["llamaindex"] = dict(imports=[p1, "from llama_index.core.schema import NodeWithScore, TextNode"],
                             top=[(n.node.get_content(), round(float(n.score), 4)) for n in out])
except Exception:
    res["llamaindex"] = dict(error=traceback.format_exc()[-600:])

try:   # Haystack
    R, p1 = first([("haystack_integrations.components.rankers.sentence_transformers", "SentenceTransformersSimilarityRanker"),   # Haystack 3
                   ("haystack.components.rankers", "SentenceTransformersSimilarityRanker"),
                   ("haystack.components.rankers", "TransformersSimilarityRanker")])
    from haystack import Document
    r = R(model=M, top_k=2)
    if hasattr(r, "warm_up"):
        r.warm_up()
    out = r.run(query=Q, documents=[Document(content=p) for p in P])["documents"]
    res["haystack"] = dict(imports=[p1, "from haystack import Document"], top=[(d.content, round(float(d.score), 4)) for d in out])
except Exception:
    res["haystack"] = dict(error=traceback.format_exc()[-600:])
    try:
        import haystack.components.rankers as hr
        res["haystack"]["rankers_module"] = [n for n in dir(hr) if not n.startswith("_")]
    except Exception:
        res["haystack"]["rankers_import_error"] = traceback.format_exc()[-800:]

import importlib.metadata as md
res["versions"] = {p: (md.version(p) if p in {d.metadata["Name"].lower() for d in md.distributions()} else None)
                   for p in ["langchain", "langchain-community", "langchain-classic", "llama-index-core", "haystack-ai", "sentence-transformers-haystack", "sentence-transformers", "transformers", "torch"]}
print(json.dumps(res, indent=1, ensure_ascii=False))
