"""Fine-tune MMS-1B (CTC) on a track. Mirrors low-resource-asr/aft_mms.py.

Primary approach for id_jv and a strong contender for es_nah. Pair with a bilingual KenLM at
decode time (see lit.lm.kenlm_build + models.mms_infer).
STUB: implement CTC trainer; build per-track vocab from normalized transcripts.
"""

from __future__ import annotations


def train(track_cfg: dict, manifest_path: str, out_dir: str) -> None:
    raise NotImplementedError("aft_mms.train: implement CTC fine-tuning loop")
