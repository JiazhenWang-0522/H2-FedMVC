from base_gate import BaseGate

import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveGate(BaseGate):

    def __init__(self, d_model, num_expert, world_size, top_k=2, gate_bias=True):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert, bias = gate_bias)
        self.top_k = top_k

    def forward(self, inp, return_all_scores=False):
        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        # dummy loss
        self.set_loss(torch.zeros(1, requires_grad=True).to(inp.device))

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score