"""Add LICENSE (Apache-2.0 text) and NOTICE (base-model attribution incl. its MIT notice) to every Horizon-Labs model repo.
python release/add_license_notice.py APACHE_TXT [--only repo1,repo2]"""
import sys
from huggingface_hub import HfApi, CommitOperationAdd
api = HfApi(); apache = open(sys.argv[1]).read()
only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[2] == "--only" else None
BASE = {"jhu-clsp/mmBERT-small": ("mmBERT-small", "the mmBERT authors (Johns Hopkins University CLSP)"),
        "jhu-clsp/mmBERT-base": ("mmBERT-base", "the mmBERT authors (Johns Hopkins University CLSP)"),
        "BAAI/bge-m3": ("BGE-M3", "the BGE-M3 authors (Beijing Academy of Artificial Intelligence)")}
MIT = """Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""
out = {}
for m in api.list_models(author="Horizon-Labs"):
    if only and m.id not in only:
        continue
    i = api.model_info(m.id); cd = i.card_data.to_dict() if i.card_data else {}
    assert cd.get("license") == "apache-2.0", m.id
    b = cd.get("base_model"); name, holder = BASE[b]
    notice = f"""{m.id.split('/')[1]}
Copyright 2026 Horizon Labs

This model is licensed under the Apache License, Version 2.0 (see LICENSE).

It is a fine-tuned version of {name} (https://huggingface.co/{b}), which is licensed under the MIT License. The model weights
and the tokenizer files derive from {name}; its license notice follows.

  Copyright (c) {holder}

{chr(10).join('  ' + l if l else '' for l in MIT.splitlines())}

Training data sources and their licenses are listed in the "Training" section of README.md.
"""
    ops = [CommitOperationAdd("LICENSE", apache.encode()), CommitOperationAdd("NOTICE", notice.encode())]
    out[m.id] = api.create_commit(m.id, operations=ops, commit_message="Add LICENSE (Apache-2.0) and NOTICE (base model attribution)").oid
    print(m.id, out[m.id][:8], flush=True)
