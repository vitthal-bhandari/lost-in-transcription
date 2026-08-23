"""Fine-tune XLS-R (CTC) on a track. Mirrors low-resource-asr/aft_xlsr.py.

Alternative CTC backbone to MMS for es_nah; useful for ablations and ensembling.
STUB: implement CTC trainer (largely shared with aft_mms; factor common code if it grows).
"""

from __future__ import annotations


def train(track_cfg: dict, manifest_path: str, out_dir: str) -> None:
    raise NotImplementedError("aft_xlsr.train: implement CTC fine-tuning loop")
