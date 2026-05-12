from __future__ import annotations

from typing import Dict

import torch

from llm_rl_final_proj.rl.base import RLAlgorithm
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch, iter_minibatches
from llm_rl_final_proj.models.logprobs import (
    approx_kl_from_logprobs,
    compute_per_token_logprobs,
    masked_mean,
    masked_mean_per_row,
)


class GSPO(RLAlgorithm):
    """Sequence-level clipped surrogate using geometric-mean likelihood ratios."""

    name = "gspo"

    def update(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        grad_accum_steps: int = 1,
    ) -> Dict[str, float]:
        # TODO(student): implement GSPO.
        # The main change relative to GRPO is that you should aggregate token log-ratios into
        # one sequence-level ratio before applying PPO-style clipping.
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
                token_log_ratio = (new_logprobs - mb.old_logprobs).clamp(-20.0, 20.0)
                mask = mb.completion_mask
                seq_log_ratio = (token_log_ratio * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [B]
                seq_ratio = seq_log_ratio.exp()

                adv = mb.advantages
                surr1 = seq_ratio * adv
                surr2 = seq_ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                with torch.no_grad():
                    clipped = (seq_ratio - 1.0).abs() > cfg.clip_eps
                    clip_frac = clipped.float().mean().item()

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
            "gspo/policy_loss": total_policy_loss / n,
            "gspo/kl": total_kl / n,
            "gspo/clip_frac": total_clip_frac / n,
            "gspo/grad_norm": total_grad_norm / n,
        }
