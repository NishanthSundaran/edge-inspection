"""Fit PatchCore on a 6 GB GPU.

anomalib keeps every training patch embedding on the GPU and grows the bank with torch.cat per batch, so
the bank is copied each step: hazelnut (352 images x 1024 patches x 1536 floats = 2.2 GB) needs 4.4 GB
for the copy alone and runs out of memory. This patch accumulates the embeddings in CPU RAM and moves
them to the GPU once, for coreset selection. The selected coreset, and so the model, is unchanged.
"""

from __future__ import annotations

import torch


def patchcore_cpu_bank() -> None:
    from anomalib.models.image.patchcore import torch_model as tm
    if getattr(tm.PatchcoreModel, "_cpu_bank_patched", False):
        return
    orig_forward, orig_sub = tm.PatchcoreModel.forward, tm.PatchcoreModel.subsample_embedding

    def forward(self, input_tensor):
        if not self.training:
            return orig_forward(self, input_tensor)
        self.memory_bank = torch.empty(0, device=input_tensor.device)
        emb = orig_forward(self, input_tensor)          # with an empty bank it just stores `emb`
        self.__dict__.setdefault("_cpu_chunks", []).append(emb.detach().cpu())
        self.memory_bank = torch.empty(0, device=input_tensor.device)
        return emb

    def subsample_embedding(self, sampling_ratio, embeddings=None):
        chunks = self.__dict__.pop("_cpu_chunks", None)
        if chunks:
            dev = next(self.feature_extractor.parameters()).device
            self.memory_bank = torch.cat(chunks).to(dev)
            del chunks
        orig_sub(self, sampling_ratio)
        torch.cuda.empty_cache()

    tm.PatchcoreModel.forward = forward
    tm.PatchcoreModel.subsample_embedding = subsample_embedding
    tm.PatchcoreModel._cpu_bank_patched = True
