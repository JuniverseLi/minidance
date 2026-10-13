"""Frozen text encoder: Qwen3-0.6B hidden states (multilingual, so Chinese prompts work too)."""
import torch

TEXT_ENCODER_REPO = "Qwen/Qwen3-0.6B"


class TextEncoder:
    def __init__(self, repo=TEXT_ENCODER_REPO, device="cuda", max_len=128):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo, dtype=torch.bfloat16).to(device).eval()
        self.model.requires_grad_(False)
        self.device, self.max_len = device, max_len
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def __call__(self, texts):
        """list[str] -> features [B, L, dim] (bf16), mask [B, L] (True = real token).

        An end-of-text token is appended so even an empty prompt (used for CFG) has one token.
        """
        texts = [t + self.tok.eos_token for t in texts]
        x = self.tok(texts, padding="longest", truncation=True, max_length=self.max_len,
                     return_tensors="pt").to(self.device)
        h = self.model(**x).last_hidden_state
        return h, x.attention_mask.bool()
