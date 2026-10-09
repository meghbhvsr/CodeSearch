"""Save a Hub model as a local sentence-transformers model with safetensors weights.

microsoft/unixcoder-base only publishes pytorch_model.bin, and transformers refuses to
torch.load .bin files on torch < 2.6 (CVE-2025-32434). This loads Microsoft's official
weights once with torch.load(weights_only=True) and re-saves them as safetensors, so
both eval and training can start from models/unixcoder-base.

  python scripts/convert_base_model.py
  python scripts/convert_base_model.py --model microsoft/codebert-base --out models/codebert-base
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("USE_TF", "0")  # a broken TensorFlow install otherwise breaks transformers imports

import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from tokenizers import processors  # noqa: E402
from sentence_transformers import SentenceTransformer, models  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="microsoft/unixcoder-base")
ap.add_argument("--out", default=str(ROOT / "models" / "unixcoder-base"))
args = ap.parse_args()

config = AutoConfig.from_pretrained(args.model)
state = torch.load(hf_hub_download(args.model, "pytorch_model.bin"), map_location="cpu", weights_only=True)
hf_model = AutoModel.from_config(config)
missing, unexpected = hf_model.load_state_dict(state, strict=False)
# The .bin may carry an LM head / pooler the bare encoder doesn't use; anything *missing* is a real problem.
if [k for k in missing if "pooler" not in k]:
    sys.exit(f"missing weights: {missing}")
print(f"loaded {args.model} (ignored {len(unexpected)} unused tensors, {len(missing)} missing pooler tensors)")

out = Path(args.out)
hf_model.save_pretrained(out, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(args.model)
mode_id = tokenizer.convert_tokens_to_ids("<encoder-only>")
if mode_id != tokenizer.unk_token_id:
    # UniXcoder is one model with three modes, picked by a token after <s>. Retrieval uses
    # encoder-only: <s> <encoder-only> </s> tokens </s>. Without it, zero-shot MRR@10 on a
    # 3k test subset drops from 0.571 to 0.450. Baking the token into the post-processor
    # means sentence-transformers adds it on every encode, both in eval and in training.
    cls, sep = tokenizer.cls_token, tokenizer.sep_token
    tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{cls} <encoder-only> {sep} $A {sep}",
        pair=f"{cls} <encoder-only> {sep} $A {sep} $B {sep}",
        special_tokens=[(cls, tokenizer.cls_token_id), ("<encoder-only>", mode_id), (sep, tokenizer.sep_token_id)],
    )
    print("added UniXcoder <encoder-only> mode prefix to the tokenizer")
tokenizer.save_pretrained(out)

transformer = models.Transformer(str(out), max_seq_length=256)
pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
SentenceTransformer(modules=[transformer, pooling]).save(str(out), safe_serialization=True)
print(f"saved -> {out}")
