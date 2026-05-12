from __future__ import annotations

from typing import Dict

from llm_rl_final_proj.models.logprobs import approx_kl_from_logprobs, compute_per_token_logprobs, masked_mean, masked_mean_per_row
import torch

from llm_rl_final_proj.rl.base import RLAlgorithm
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch, iter_minibatches


class DrGRPO(RLAlgorithm):
    """DrGRPO removes the GRPO sequence-length normalization and std scaling."""

    name = "dr_grpo"

    def update(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        grad_accum_steps: int = 1,
    ) -> Dict[str, float]:
        # TODO(student): implement DrGRPO.
        # Start from your GRPO implementation, then make the two intended changes:
        #   1. use the DrGRPO advantage convention (configured in online/train_rm_grpo.py),
        #   2. remove the per-sequence length normalization inside the surrogate.
        cfg = self.cfg
        device = next(model.parameters()).device
        total_policy_loss = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        total_grad_norm = 0.0
        n_updates = 0
        model.train()

        for epoch in range(cfg.ppo_epochs):
            for mb in iter_minibatches(rollout, minibatch_size=cfg.minibatch_size, shuffle=True, device=device,):
                new_logprobs = compute_per_token_logprobs(model, mb.input_ids, mb.attention_mask, enable_grad=True,)
                log_ratio = new_logprobs - mb.old_logprobs
                ratio = log_ratio.exp()
                adv = mb.advantages.unsqueeze(1)
                surr1 = ratio * adv
                surr2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
                per_token_obj = torch.min(surr1, surr2)       # [B, L-1]
                policy_loss = -(per_token_obj * mb.completion_mask).sum(dim=1).mean()
                
                with torch.no_grad():
                    clipped = (ratio - 1.0).abs() > cfg.clip_eps
                    clip_frac = masked_mean(clipped.float(), mb.completion_mask).item()

                kl = approx_kl_from_logprobs(
                    new_logprobs, mb.ref_logprobs, mb.completion_mask
                )

                
                loss = policy_loss + cfg.kl_coef * kl
                (loss / grad_accum_steps).backward()
                n_updates += 1

                if n_updates % grad_accum_steps == 0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.max_grad_norm
                        ).item()
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    grad_norm = 0.0

                total_policy_loss += policy_loss.item()
                total_kl += kl.item()
                total_clip_frac += clip_frac
                total_grad_norm += grad_norm

        if n_updates % grad_accum_steps != 0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm
                ).item()
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_grad_norm += grad_norm

        n = max(1, n_updates)
        return {
            "dr_grpo/policy_loss": total_policy_loss / n,
            "dr_grpo/kl": total_kl / n,
            "dr_grpo/clip_frac": total_clip_frac / n,
            "dr_grpo/grad_norm": total_grad_norm / n,
        }
