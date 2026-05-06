import torch
import torch.nn as nn
import torch.nn.functional as F

from naive_gate import NaiveGate

class FedMoE(nn.Module):

    def __init__(self, client_models, feature_dim, num_classes, top_k=3, device="cuda"):
        super(FedMoE, self).__init__()
        self.device = device
        self.client_models = nn.ModuleList(client_models)
        self.num_clients = len(client_models)
        self.ln = nn.LayerNorm(feature_dim)
        self.gate = NaiveGate(
            d_model=feature_dim,
            num_expert=self.num_clients,
            world_size=1,
            top_k=top_k,
            gate_bias=True
        ).to(device)

        self.classifier = nn.Linear(feature_dim, num_classes).to(device)

        for m in self.client_models:
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, xs):
        client_hs = []
        with torch.no_grad():
            for model in self.client_models:
                _, _, h, _ = model(xs)
                client_hs.append(h.unsqueeze(1))


        client_hs = torch.cat(client_hs, dim=1)
        token_feat = client_hs.mean(dim=1)  # [B, F]
        token_feat = self.ln(token_feat)
        gate_idx, gate_score = self.gate(token_feat)

        B, top_k = gate_idx.size()
        Fdim = client_hs.size(-1)

        gather_index = gate_idx.unsqueeze(-1).expand(-1, -1, Fdim)
        gathered_h = client_hs.gather(dim=1, index=gather_index)
        weights = gate_score.unsqueeze(-1)
        mixed_h = (weights * gathered_h).sum(dim=1)
        logits = self.classifier(mixed_h)
        return logits, gate_idx, gate_score, mixed_h
