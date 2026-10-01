#!/bin/bash
# usage: M=<release dir> bash rerank/run_st_check.sh  (sentence-transformers CrossEncoder smoke test of the card snippet)
source scripts/env.sh
python - <<PY
from sentence_transformers import CrossEncoder
m = CrossEncoder("$M", device="cuda")
q = "How tall is the Eiffel Tower?"
ps = ["The Eiffel Tower is 330 metres tall.", "La tour Eiffel a été construite pour l'Exposition universelle de 1889.", "The Statue of Liberty is 93 metres tall."]
print(m.rank(q, ps)); print(m.predict([(q, p) for p in ps]))
PY
