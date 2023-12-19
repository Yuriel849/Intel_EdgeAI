from pathlib import Path
from bark.generation import load_model, codec_decode, _flatten_codebooks
import torch
import openvino as ov


class TextEncoderModel(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, idx, past_kv=None):
        return self.encoder(idx, merge_context=True, past_kv=past_kv, use_cache=True)


class CoarseEncoderModel(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, idx, past_kv=None):
        return self.encoder(idx, past_kv=past_kv, use_cache=True)


class FineModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pred_idx, idx):
        b, t, codes = idx.size()
        pos = torch.arange(0, t, dtype=torch.long).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_embs = [
            wte(idx[:, :, i]).unsqueeze(-1)
            for i, wte in enumerate(self.model.transformer.wtes)
        ]  # token embeddings of shape (b, t, n_embd)
        tok_emb = torch.cat(tok_embs, dim=-1)
        pos_emb = self.model.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = tok_emb[:, :, :, : pred_idx + 1].sum(dim=-1)
        x = self.model.transformer.drop(x + pos_emb)
        for block in self.model.transformer.h:
            x = block(x)
        x = self.model.transformer.ln_f(x)
        return x
