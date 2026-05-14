from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

from llm_rl_final_proj.data.ultrafeedback import GenerationExample, build_generation_examples, dataset_overview
from llm_rl_final_proj.data.ultrafeedback import build_preference_examples
from llm_rl_final_proj.models.logprobs import compute_per_token_logprobs
from llm_rl_final_proj.models.load import (
    load_lora_policy_model_and_tokenizer,
    load_reward_model_and_tokenizer,
)
from llm_rl_final_proj.offline.evaluation import generate_samples, summarize_generation_rows
from llm_rl_final_proj.reward_model.evaluation import score_prompt_response_pairs
from llm_rl_final_proj.reward_model.evaluation import evaluate_reward_model_dataset
from llm_rl_final_proj.rl.base import AlgoConfig
from llm_rl_final_proj.rl.dr_grpo import DrGRPO
from llm_rl_final_proj.rl.gspo import GSPO
from llm_rl_final_proj.rl.grpo import GRPO
from llm_rl_final_proj.rl.reinforce import Reinforce
from llm_rl_final_proj.rollout.hf_sampler import HFSampler, SamplingConfig
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch
from llm_rl_final_proj.utils.hardware import (
    get_cuda_memory_metrics,
    get_hardware_metrics,
    get_model_device_metrics,
    require_cuda_if_requested,
    resolve_device_and_dtype,
)
from llm_rl_final_proj.utils.seed import set_seed
from llm_rl_final_proj.utils.wandb_utils import WandBLogger


@dataclass
class OnlineRMGRPOConfig:
    algo: str = "grpo"
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    reward_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    reward_adapter_path: str = ""
    reward_adapter_paths: List[str] | None = None
    reward_model_select_best: bool = False
    reward_selection_split: str = "test_prefs"
    reward_selection_limit: int = 256
    reward_aggregation: str = "single"
    reward_pessimism_coef: float = 1.0
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    train_split: str = "train_gen"
    eval_split: str = "test_gen"
    output_dir: str = "runs/rm_grpo_default"

    seed: int = 0
    steps: int = 101
    batch_size: int = 8
    group_size: int = 4

    min_new_tokens: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0

    lr: float = 3e-5
    weight_decay: float = 0.0
    betas1: float = 0.9
    betas2: float = 0.95
    warmup_steps: int = 20
    grad_accum_steps: int = 1
    max_grad_norm: float = 0.5

    ppo_epochs: int = 2
    minibatch_size: int = 8
    clip_eps: float = 0.1
    clip_eps_high: float = 0.0
    kl_coef: float = 0.02
    adv_clip: float = 5.0
    advantage_mode: str = "reward"

    max_prompt_tokens: int = 700
    max_response_tokens: int = 256
    train_limit: int = 0
    eval_limit: int = 64
    reward_batch_size: int = 16

    eval_interval: int = 25
    save_interval: int = 50
    eval_max_new_tokens: int = 256
    eval_temperature: float = 0.0
    eval_top_p: float = 1.0
    eval_batch_size: int = 8

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_bias: str = "none"
    grad_checkpointing: bool = True

    wandb_project: str = "llm-rl-final-project"
    wandb_name: str = "rm_grpo"
    wandb_enabled: bool = True
    sample_log_n: int = 8
    sample_log_max_chars: int = 2500

    replay_enabled: bool = False
    replay_capacity: int = 128
    replay_batch_size: int = 8
    replay_updates_per_step: int = 1
    replay_loss_weight: float = 1.0
    replay_algo: str = "dpo"
    replay_beta: float = 0.1
    replay_min_reward_gap: float = 0.0


@dataclass
class RewardModelHandle:
    adapter_path: str
    model: torch.nn.Module
    tokenizer: Any


@dataclass
class ReplayPreferenceExample:
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    chosen_completion_mask: torch.Tensor
    chosen_ref_logprobs: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor
    rejected_completion_mask: torch.Tensor
    rejected_ref_logprobs: torch.Tensor
    reward_margin: float


def parse_args() -> OnlineRMGRPOConfig:
    ap = argparse.ArgumentParser(description="Train a policy online with a GRPO-family algorithm using a learned reward model.")
    ap.add_argument(
        "--algo",
        type=str,
        default=OnlineRMGRPOConfig.algo,
        choices=["grpo", "dr_grpo", "gspo"],
    )
    ap.add_argument("--model_name", type=str, default=OnlineRMGRPOConfig.model_name)
    ap.add_argument("--reward_model_name", type=str, default=OnlineRMGRPOConfig.reward_model_name)
    ap.add_argument("--reward_adapter_path", type=str, default=OnlineRMGRPOConfig.reward_adapter_path)
    ap.add_argument("--reward_adapter_paths", type=str, nargs="+", default=None)
    ap.add_argument(
        "--reward_model_select_best",
        action=argparse.BooleanOptionalAction,
        default=OnlineRMGRPOConfig.reward_model_select_best,
    )
    ap.add_argument("--reward_selection_split", type=str, default=OnlineRMGRPOConfig.reward_selection_split)
    ap.add_argument("--reward_selection_limit", type=int, default=OnlineRMGRPOConfig.reward_selection_limit)
    ap.add_argument(
        "--reward_aggregation",
        type=str,
        default=OnlineRMGRPOConfig.reward_aggregation,
        choices=["single", "mean", "min", "pessimistic"],
    )
    ap.add_argument("--reward_pessimism_coef", type=float, default=OnlineRMGRPOConfig.reward_pessimism_coef)
    ap.add_argument("--dataset_name", type=str, default=OnlineRMGRPOConfig.dataset_name)
    ap.add_argument("--train_split", type=str, default=OnlineRMGRPOConfig.train_split)
    ap.add_argument("--eval_split", type=str, default=OnlineRMGRPOConfig.eval_split)
    ap.add_argument("--output_dir", type=str, default=OnlineRMGRPOConfig.output_dir)

    ap.add_argument("--seed", type=int, default=OnlineRMGRPOConfig.seed)
    ap.add_argument("--steps", type=int, default=OnlineRMGRPOConfig.steps)
    ap.add_argument("--batch_size", type=int, default=OnlineRMGRPOConfig.batch_size)
    ap.add_argument("--group_size", type=int, default=OnlineRMGRPOConfig.group_size)

    ap.add_argument("--min_new_tokens", type=int, default=OnlineRMGRPOConfig.min_new_tokens)
    ap.add_argument("--max_new_tokens", type=int, default=OnlineRMGRPOConfig.max_new_tokens)
    ap.add_argument("--temperature", type=float, default=OnlineRMGRPOConfig.temperature)
    ap.add_argument("--top_p", type=float, default=OnlineRMGRPOConfig.top_p)
    ap.add_argument("--top_k", type=int, default=OnlineRMGRPOConfig.top_k)
    ap.add_argument("--repetition_penalty", type=float, default=OnlineRMGRPOConfig.repetition_penalty)

    ap.add_argument("--lr", type=float, default=OnlineRMGRPOConfig.lr)
    ap.add_argument("--weight_decay", type=float, default=OnlineRMGRPOConfig.weight_decay)
    ap.add_argument("--betas1", type=float, default=OnlineRMGRPOConfig.betas1)
    ap.add_argument("--betas2", type=float, default=OnlineRMGRPOConfig.betas2)
    ap.add_argument("--warmup_steps", type=int, default=OnlineRMGRPOConfig.warmup_steps)
    ap.add_argument("--grad_accum_steps", type=int, default=OnlineRMGRPOConfig.grad_accum_steps)
    ap.add_argument("--max_grad_norm", type=float, default=OnlineRMGRPOConfig.max_grad_norm)

    ap.add_argument("--ppo_epochs", type=int, default=OnlineRMGRPOConfig.ppo_epochs)
    ap.add_argument("--minibatch_size", type=int, default=OnlineRMGRPOConfig.minibatch_size)
    ap.add_argument("--clip_eps", type=float, default=OnlineRMGRPOConfig.clip_eps)
    ap.add_argument("--clip_eps_high", type=float, default=OnlineRMGRPOConfig.clip_eps_high)
    ap.add_argument("--kl_coef", type=float, default=OnlineRMGRPOConfig.kl_coef)
    ap.add_argument("--adv_clip", type=float, default=OnlineRMGRPOConfig.adv_clip)
    ap.add_argument(
        "--advantage_mode",
        type=str,
        default=OnlineRMGRPOConfig.advantage_mode,
        choices=["reward", "rank"],
    )

    ap.add_argument("--max_prompt_tokens", type=int, default=OnlineRMGRPOConfig.max_prompt_tokens)
    ap.add_argument("--max_response_tokens", type=int, default=OnlineRMGRPOConfig.max_response_tokens)
    ap.add_argument("--train_limit", type=int, default=OnlineRMGRPOConfig.train_limit)
    ap.add_argument("--eval_limit", type=int, default=OnlineRMGRPOConfig.eval_limit)
    ap.add_argument("--reward_batch_size", type=int, default=OnlineRMGRPOConfig.reward_batch_size)

    ap.add_argument("--eval_interval", type=int, default=OnlineRMGRPOConfig.eval_interval)
    ap.add_argument("--save_interval", type=int, default=OnlineRMGRPOConfig.save_interval)
    ap.add_argument("--eval_max_new_tokens", type=int, default=OnlineRMGRPOConfig.eval_max_new_tokens)
    ap.add_argument("--eval_temperature", type=float, default=OnlineRMGRPOConfig.eval_temperature)
    ap.add_argument("--eval_top_p", type=float, default=OnlineRMGRPOConfig.eval_top_p)
    ap.add_argument("--eval_batch_size", type=int, default=OnlineRMGRPOConfig.eval_batch_size)

    ap.add_argument("--lora_r", type=int, default=OnlineRMGRPOConfig.lora_r)
    ap.add_argument("--lora_alpha", type=int, default=OnlineRMGRPOConfig.lora_alpha)
    ap.add_argument("--lora_dropout", type=float, default=OnlineRMGRPOConfig.lora_dropout)
    ap.add_argument("--lora_target_modules", type=str, default=OnlineRMGRPOConfig.lora_target_modules)
    ap.add_argument("--lora_bias", type=str, default=OnlineRMGRPOConfig.lora_bias)
    ap.add_argument(
        "--grad_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=OnlineRMGRPOConfig.grad_checkpointing,
    )

    ap.add_argument("--wandb_project", type=str, default=OnlineRMGRPOConfig.wandb_project)
    ap.add_argument("--wandb_name", type=str, default=OnlineRMGRPOConfig.wandb_name)
    ap.add_argument(
        "--wandb_enabled",
        action=argparse.BooleanOptionalAction,
        default=OnlineRMGRPOConfig.wandb_enabled,
    )
    ap.add_argument("--sample_log_n", type=int, default=OnlineRMGRPOConfig.sample_log_n)
    ap.add_argument("--sample_log_max_chars", type=int, default=OnlineRMGRPOConfig.sample_log_max_chars)
    ap.add_argument(
        "--replay_enabled",
        action=argparse.BooleanOptionalAction,
        default=OnlineRMGRPOConfig.replay_enabled,
    )
    ap.add_argument("--replay_capacity", type=int, default=OnlineRMGRPOConfig.replay_capacity)
    ap.add_argument("--replay_batch_size", type=int, default=OnlineRMGRPOConfig.replay_batch_size)
    ap.add_argument("--replay_updates_per_step", type=int, default=OnlineRMGRPOConfig.replay_updates_per_step)
    ap.add_argument("--replay_loss_weight", type=float, default=OnlineRMGRPOConfig.replay_loss_weight)
    ap.add_argument("--replay_algo", type=str, default=OnlineRMGRPOConfig.replay_algo, choices=["dpo", "ipo"])
    ap.add_argument("--replay_beta", type=float, default=OnlineRMGRPOConfig.replay_beta)
    ap.add_argument("--replay_min_reward_gap", type=float, default=OnlineRMGRPOConfig.replay_min_reward_gap)
    args = ap.parse_args()
    return OnlineRMGRPOConfig(**vars(args))


def maybe_update_warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int) -> None:
    if warmup_steps <= 0:
        scale = 1.0
    else:
        scale = min(1.0, float(step + 1) / float(warmup_steps))
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * scale


def _normalize_lora_target_modules(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _sample_prompt_batch(examples: Sequence[GenerationExample], batch_size: int, rng: random.Random) -> List[GenerationExample]:
    if not examples:
        raise RuntimeError("Cannot sample prompts from an empty generation split.")
    return [examples[rng.randrange(len(examples))] for _ in range(batch_size)]


def _compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
    *,
    divide_by_std: bool,
) -> torch.Tensor:
    # TODO(student): compute one scalar advantage per sampled completion by grouping rewards
    # into prompt-wise batches of size `group_size`, subtracting the group mean, and optionally
    # dividing by the group standard deviation when `divide_by_std=True`.
    n = rewards.shape[0]
    rewards = rewards.view(-1, group_size)
    mean = rewards.mean(dim = 1, keepdim=True)
    advantage = rewards - mean
    if divide_by_std:
        std = rewards.std(dim=1, keepdim=True, unbiased=False)
        advantage = advantage / (std + eps)
    return advantage.view(-1)


def _compute_rank_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    grouped = rewards.view(-1, group_size)
    order = torch.argsort(grouped, dim=1)
    ranks = torch.empty_like(grouped)
    rank_values = torch.arange(group_size, device=rewards.device, dtype=rewards.dtype)
    ranks.scatter_(1, order, rank_values.expand_as(grouped))
    centered = ranks - ranks.mean(dim=1, keepdim=True)
    scale = centered.std(dim=1, keepdim=True, unbiased=False)
    return (centered / (scale + eps)).view(-1)


def _compute_advantages(
    rewards: torch.Tensor,
    group_size: int,
    *,
    advantage_mode: str,
    divide_by_std: bool,
) -> torch.Tensor:
    if advantage_mode == "rank":
        return _compute_rank_advantages(rewards, group_size)
    if advantage_mode == "reward":
        return _compute_group_advantages(rewards, group_size, divide_by_std=divide_by_std)
    raise ValueError(f"Unsupported advantage_mode={advantage_mode}")


def _resolve_reward_adapter_paths(cfg: OnlineRMGRPOConfig) -> List[str]:
    paths = list(cfg.reward_adapter_paths or [])
    if cfg.reward_adapter_path:
        paths.insert(0, cfg.reward_adapter_path)
    paths = [str(p) for p in paths if str(p).strip()]
    deduped = list(dict.fromkeys(paths))
    if not deduped:
        raise ValueError("Provide --reward_adapter_path or --reward_adapter_paths.")
    if cfg.reward_aggregation != "single" and len(deduped) < 2:
        raise ValueError(f"--reward_aggregation {cfg.reward_aggregation!r} requires at least two reward adapters.")
    return deduped


def _load_reward_handle(
    *,
    adapter_path: str,
    cfg: OnlineRMGRPOConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> RewardModelHandle:
    loaded = load_reward_model_and_tokenizer(
        cfg.reward_model_name,
        device=device,
        dtype=dtype,
        adapter_path=adapter_path,
    )
    loaded.model.eval()
    for p in loaded.model.parameters():
        p.requires_grad_(False)
    return RewardModelHandle(adapter_path=adapter_path, model=loaded.model, tokenizer=loaded.tokenizer)


def _select_reward_adapters(cfg: OnlineRMGRPOConfig, paths: Sequence[str], device: torch.device, dtype: torch.dtype) -> List[str]:
    if not cfg.reward_model_select_best:
        return list(paths)
    examples = build_preference_examples(
        cfg.dataset_name,
        cfg.reward_selection_split,
        limit=cfg.reward_selection_limit,
    )
    if not examples:
        raise RuntimeError(f"Reward-model selection split {cfg.reward_selection_split!r} produced zero examples.")
    scored: List[tuple[float, str]] = []
    for path in paths:
        handle = _load_reward_handle(adapter_path=path, cfg=cfg, device=device, dtype=dtype)
        metrics = evaluate_reward_model_dataset(
            handle.model,
            handle.tokenizer,
            examples,
            max_prompt_tokens=cfg.max_prompt_tokens,
            max_response_tokens=cfg.max_response_tokens,
            per_device_eval_batch_size=cfg.reward_batch_size,
            device=device,
            desc=f"select[reward_model|{Path(path).parent.name}]",
        )
        acc = float(metrics["eval/rm_pair_accuracy"])
        scored.append((acc, path))
        print(f"[reward_model_selection] adapter={path} pair_accuracy={acc:.4f}")
        del handle
        if device.type == "cuda":
            torch.cuda.empty_cache()
    scored.sort(key=lambda x: x[0], reverse=True)
    best_acc, best_path = scored[0]
    print(f"[reward_model_selection] selected={best_path} pair_accuracy={best_acc:.4f}")
    return [best_path]


@torch.no_grad()
def _score_rows_with_reward_models(
    handles: Sequence[RewardModelHandle],
    rows: Sequence[Dict[str, object]],
    *,
    cfg: OnlineRMGRPOConfig,
    device: torch.device,
) -> tuple[torch.Tensor, Dict[str, float]]:
    per_model_scores: List[List[float]] = []
    for handle in handles:
        per_model_scores.append(
            score_prompt_response_pairs(
                handle.model,
                handle.tokenizer,
                rows,
                max_prompt_tokens=cfg.max_prompt_tokens,
                max_response_tokens=cfg.max_response_tokens,
                per_device_batch_size=cfg.reward_batch_size,
                device=device,
            )
        )
    score_matrix = torch.tensor(per_model_scores, device=device, dtype=torch.float32)
    if cfg.reward_aggregation == "single" or score_matrix.shape[0] == 1:
        rewards = score_matrix[0]
    elif cfg.reward_aggregation == "mean":
        rewards = score_matrix.mean(dim=0)
    elif cfg.reward_aggregation == "min":
        rewards = score_matrix.min(dim=0).values
    elif cfg.reward_aggregation == "pessimistic":
        rewards = score_matrix.mean(dim=0) - cfg.reward_pessimism_coef * score_matrix.std(dim=0, unbiased=False)
    else:
        raise ValueError(f"Unsupported reward_aggregation={cfg.reward_aggregation}")
    metrics = {
        "reward_ensemble/model_count": float(score_matrix.shape[0]),
        "reward_ensemble/aggregate_mean": float(rewards.mean().item()),
        "reward_ensemble/aggregate_std": float(rewards.std(unbiased=False).item()),
    }
    if score_matrix.shape[0] > 1:
        per_item_std = score_matrix.std(dim=0, unbiased=False)
        metrics["reward_ensemble/member_score_std_mean"] = float(per_item_std.mean().item())
        metrics["reward_ensemble/member_score_range_mean"] = float((score_matrix.max(dim=0).values - score_matrix.min(dim=0).values).mean().item())
    return rewards, metrics


def _add_rollout_preferences_to_replay(
    replay: deque[ReplayPreferenceExample],
    rollout,
    rewards: torch.Tensor,
    *,
    min_reward_gap: float,
) -> int:
    added = 0
    group_size = int(rollout.group_size)
    reward_groups = rewards.detach().cpu().view(-1, group_size)
    for group_idx, group_rewards in enumerate(reward_groups):
        best_local = int(torch.argmax(group_rewards).item())
        worst_local = int(torch.argmin(group_rewards).item())
        reward_gap = float((group_rewards[best_local] - group_rewards[worst_local]).item())
        if best_local == worst_local or reward_gap < min_reward_gap:
            continue
        best = group_idx * group_size + best_local
        worst = group_idx * group_size + worst_local
        replay.append(
            ReplayPreferenceExample(
                chosen_input_ids=rollout.input_ids[best].detach().cpu(),
                chosen_attention_mask=rollout.attention_mask[best].detach().cpu(),
                chosen_completion_mask=rollout.completion_mask[best].detach().cpu(),
                chosen_ref_logprobs=rollout.ref_logprobs[best].detach().cpu(),
                rejected_input_ids=rollout.input_ids[worst].detach().cpu(),
                rejected_attention_mask=rollout.attention_mask[worst].detach().cpu(),
                rejected_completion_mask=rollout.completion_mask[worst].detach().cpu(),
                rejected_ref_logprobs=rollout.ref_logprobs[worst].detach().cpu(),
                reward_margin=reward_gap,
            )
        )
        added += 1
    return added


def _pad_1d_tensors(
    rows: Sequence[torch.Tensor],
    *,
    pad_value: float | int,
    dtype: torch.dtype,
    max_len: int | None = None,
) -> torch.Tensor:
    if max_len is None:
        max_len = max(int(x.numel()) for x in rows)
    out = torch.full((len(rows), max_len), pad_value, dtype=dtype)
    for i, row in enumerate(rows):
        n = int(row.numel())
        out[i, :n] = row.to(dtype=dtype)
    return out


def _masked_sum_per_row(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum(dim=1)


def _replay_preference_update(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: deque[ReplayPreferenceExample],
    cfg: OnlineRMGRPOConfig,
    tokenizer,
    device: torch.device,
    rng: random.Random,
) -> Dict[str, float]:
    if not cfg.replay_enabled or len(replay) == 0 or cfg.replay_updates_per_step <= 0:
        return {}
    model.train()
    total_loss = 0.0
    total_margin = 0.0
    total_reward_margin = 0.0
    n_updates = 0
    batch_size = min(cfg.replay_batch_size, len(replay))
    pad_id = int(tokenizer.pad_token_id)

    for _ in range(cfg.replay_updates_per_step):
        examples = rng.sample(list(replay), batch_size)
        seq_max_len = max(
            max(int(ex.chosen_input_ids.numel()), int(ex.rejected_input_ids.numel()))
            for ex in examples
        )
        logprob_max_len = seq_max_len - 1
        chosen_input_ids = _pad_1d_tensors([ex.chosen_input_ids for ex in examples], pad_value=pad_id, dtype=torch.long, max_len=seq_max_len).to(device)
        rejected_input_ids = _pad_1d_tensors([ex.rejected_input_ids for ex in examples], pad_value=pad_id, dtype=torch.long, max_len=seq_max_len).to(device)
        chosen_attention_mask = _pad_1d_tensors([ex.chosen_attention_mask for ex in examples], pad_value=0, dtype=torch.long, max_len=seq_max_len).to(device)
        rejected_attention_mask = _pad_1d_tensors([ex.rejected_attention_mask for ex in examples], pad_value=0, dtype=torch.long, max_len=seq_max_len).to(device)
        chosen_completion_mask = _pad_1d_tensors([ex.chosen_completion_mask for ex in examples], pad_value=0.0, dtype=torch.float32, max_len=logprob_max_len).to(device)
        rejected_completion_mask = _pad_1d_tensors([ex.rejected_completion_mask for ex in examples], pad_value=0.0, dtype=torch.float32, max_len=logprob_max_len).to(device)
        chosen_ref_logprobs = _pad_1d_tensors([ex.chosen_ref_logprobs for ex in examples], pad_value=0.0, dtype=torch.float32, max_len=logprob_max_len).to(device)
        rejected_ref_logprobs = _pad_1d_tensors([ex.rejected_ref_logprobs for ex in examples], pad_value=0.0, dtype=torch.float32, max_len=logprob_max_len).to(device)

        input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
        completion_mask = torch.cat([chosen_completion_mask, rejected_completion_mask], dim=0)
        policy_logprobs = compute_per_token_logprobs(model, input_ids, attention_mask, enable_grad=True)
        chosen_policy, rejected_policy = _masked_sum_per_row(policy_logprobs, completion_mask).chunk(2, dim=0)
        chosen_ref = _masked_sum_per_row(chosen_ref_logprobs, chosen_completion_mask)
        rejected_ref = _masked_sum_per_row(rejected_ref_logprobs, rejected_completion_mask)
        logits = cfg.replay_beta * ((chosen_policy - rejected_policy) - (chosen_ref - rejected_ref))

        if cfg.replay_algo == "dpo":
            loss = -F.logsigmoid(logits).mean()
        elif cfg.replay_algo == "ipo":
            loss = ((logits - (1.0 / (2.0 * cfg.replay_beta))) ** 2).mean()
        else:
            raise ValueError(f"Unsupported replay_algo={cfg.replay_algo}")

        weighted_loss = cfg.replay_loss_weight * loss
        weighted_loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm).item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item())
        total_margin += float(logits.detach().mean().item())
        total_reward_margin += float(sum(ex.reward_margin for ex in examples) / max(1, len(examples)))
        n_updates += 1

    n = max(1, n_updates)
    return {
        "replay/loss": total_loss / n,
        "replay/reference_corrected_margin": total_margin / n,
        "replay/reward_margin": total_reward_margin / n,
        "replay/buffer_size": float(len(replay)),
        "replay/updates": float(n_updates),
        "replay/grad_norm": grad_norm if n_updates else 0.0,
    }

def _build_online_algo(cfg: OnlineRMGRPOConfig):
    algo_cfg = AlgoConfig(
        ppo_epochs=cfg.ppo_epochs,
        minibatch_size=cfg.minibatch_size,
        clip_eps=cfg.clip_eps,
        clip_eps_high=cfg.clip_eps_high,
        kl_coef=cfg.kl_coef,
        max_grad_norm=cfg.max_grad_norm,
        adv_clip=cfg.adv_clip,
        seed=cfg.seed,
    )
    if cfg.algo == "grpo":
        return GRPO(algo_cfg)
    if cfg.algo == "dr_grpo":
        return DrGRPO(algo_cfg)
    if cfg.algo == "gspo":
        return GSPO(algo_cfg)
    raise ValueError(f"Unsupported --algo {cfg.algo}")


def _algo_divides_advantages_by_std(algo: str) -> bool:
    # TODO(student): return True for the algorithms that use group-standard-deviation
    # normalization and False for the algorithms that intentionally avoid it.
    true_algs = set(["grpo"])
    if algo in true_algs:
        return True
    else:
        return False

def _normalize_completion_for_reward_scoring(text: str) -> str:
    if text.strip():
        return text
    return "[no response]"


def _truncate(text: str | None, max_chars: int) -> str | None:
    if text is None:
        return None
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[truncated]"


def _sample_rows_for_logging(
    examples: Sequence[GenerationExample],
    rows: Sequence[Dict[str, Any]],
    rm_scores: Sequence[float],
    *,
    sample_log_n: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ex, row, score in list(zip(examples, rows, rm_scores))[: max(0, sample_log_n)]:
        out.append(
            {
                "row_id": ex.row_id,
                "prompt": _truncate(ex.prompt_text, max_chars),
                "reference_response": _truncate(ex.reference_response_text, max_chars),
                "model_response": _truncate(str(row.get("model_response", "")), max_chars),
                "reward_model_score": float(score),
            }
        )
    return out


def save_checkpoint(model: torch.nn.Module, cfg: OnlineRMGRPOConfig, step: int) -> None:
    ckpt_dir = Path(cfg.output_dir) / "checkpoints" / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = ckpt_dir / "adapter"
    model.save_pretrained(adapter_dir)
    meta = {
        "step": step,
        "model_type": "online_policy_rm_rl",
        "algo": cfg.algo,
        "model_name": cfg.model_name,
        "reward_model_name": cfg.reward_model_name,
        "reward_adapter_path": cfg.reward_adapter_path,
        "reward_adapter_paths": cfg.reward_adapter_paths or [cfg.reward_adapter_path],
        "reward_aggregation": cfg.reward_aggregation,
        "advantage_mode": cfg.advantage_mode,
        "replay_enabled": cfg.replay_enabled,
        "dataset_name": cfg.dataset_name,
        "train_split": cfg.train_split,
        "eval_split": cfg.eval_split,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


@torch.no_grad()
def evaluate_policy_with_reward_model(
    *,
    policy_model: torch.nn.Module,
    policy_tokenizer,
    reward_handles: Sequence[RewardModelHandle],
    cfg: OnlineRMGRPOConfig,
    examples: Sequence[GenerationExample],
    device: torch.device,
    max_prompt_tokens: int,
    max_response_tokens: int,
    generation_max_new_tokens: int,
    temperature: float,
    top_p: float,
    generation_batch_size: int,
) -> tuple[Dict[str, float], List[Dict[str, Any]], List[float]]:
    rows = generate_samples(
        policy_model,
        policy_tokenizer,
        examples,
        device=device,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=generation_max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=generation_batch_size,
    )
    metrics = summarize_generation_rows(rows)
    scoring_rows = []
    reference_rows = []
    has_reference = True
    for ex, row in zip(examples, rows):
        scoring_rows.append(
            {
                "row_id": ex.row_id,
                "prompt_messages": ex.prompt_messages,
                "prompt_text": ex.prompt_text,
                "response_text": _normalize_completion_for_reward_scoring(str(row["model_response"])),
            }
        )
        if ex.reference_response_text:
            reference_rows.append(
                {
                    "row_id": ex.row_id,
                    "prompt_messages": ex.prompt_messages,
                    "prompt_text": ex.prompt_text,
                    "response_text": ex.reference_response_text,
                }
            )
        else:
            has_reference = False
    score_tensor, reward_metrics = _score_rows_with_reward_models(
        reward_handles,
        scoring_rows,
        cfg=cfg,
        device=device,
    )
    metrics["eval/rm_score_mean_on_policy_generations"] = float(score_tensor.mean().item())
    metrics["eval/rm_score_std_on_policy_generations"] = float(score_tensor.std(unbiased=False).item())
    metrics.update({f"eval/{k}": v for k, v in reward_metrics.items()})
    if has_reference and reference_rows:
        ref_tensor, _ = _score_rows_with_reward_models(
            reward_handles,
            reference_rows,
            cfg=cfg,
            device=device,
        )
        margin = score_tensor - ref_tensor
        metrics["eval/rm_reference_score_mean_on_dataset_reference_responses"] = float(ref_tensor.mean().item())
        metrics["eval/rm_fraction_policy_scores_above_reference"] = float((margin > 0).float().mean().item())
        metrics["eval/rm_margin_policy_minus_reference_mean"] = float(margin.mean().item())
    return metrics, rows, [float(x) for x in score_tensor.detach().cpu().tolist()]


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    require_cuda_if_requested()
    if cfg.steps <= 0:
        raise ValueError(f"--steps must be >= 1, got {cfg.steps}")
    if cfg.batch_size <= 0:
        raise ValueError(f"--batch_size must be >= 1, got {cfg.batch_size}")
    if cfg.group_size <= 0:
        raise ValueError(f"--group_size must be >= 1, got {cfg.group_size}")
    reward_adapter_paths = _resolve_reward_adapter_paths(cfg)
    if cfg.reward_model_select_best and cfg.reward_aggregation != "single":
        raise ValueError("--reward_model_select_best chooses one adapter, so use --reward_aggregation single.")
    if cfg.replay_enabled and cfg.replay_beta <= 0.0:
        raise ValueError(f"--replay_beta must be > 0, got {cfg.replay_beta}")

    if cfg.wandb_name == OnlineRMGRPOConfig.wandb_name and cfg.algo != OnlineRMGRPOConfig.algo:
        cfg.wandb_name = f"rm_{cfg.algo}"
    if cfg.output_dir == OnlineRMGRPOConfig.output_dir and cfg.algo != OnlineRMGRPOConfig.algo:
        cfg.output_dir = f"runs/rm_{cfg.algo}_default"

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_online_rm_grpo_config.json").write_text(
        json.dumps(vars(cfg), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rng = random.Random(cfg.seed)
    device, dtype = resolve_device_and_dtype()
    reward_adapter_paths = _select_reward_adapters(cfg, reward_adapter_paths, device, dtype)
    cfg.reward_adapter_path = reward_adapter_paths[0]
    cfg.reward_adapter_paths = reward_adapter_paths
    (output_dir / "resolved_online_rm_grpo_config.json").write_text(
        json.dumps(vars(cfg), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[setup] device={device} dtype={dtype} algo={cfg.algo} "
        f"policy={cfg.model_name} reward_model={cfg.reward_model_name} "
        f"reward_adapters={len(reward_adapter_paths)} reward_aggregation={cfg.reward_aggregation}"
    )
    print("[setup][hardware]", json.dumps(get_hardware_metrics(device), indent=2, sort_keys=True))

    dataset_info = dataset_overview(cfg.dataset_name)
    train_examples = build_generation_examples(cfg.dataset_name, cfg.train_split, limit=cfg.train_limit)
    eval_examples = build_generation_examples(cfg.dataset_name, cfg.eval_split, limit=cfg.eval_limit)
    if not train_examples:
        raise RuntimeError("Training generation split produced zero examples.")
    if not eval_examples:
        raise RuntimeError("Evaluation generation split produced zero examples.")

    loaded_policy = load_lora_policy_model_and_tokenizer(
        cfg.model_name,
        device=device,
        dtype=dtype,
        grad_checkpointing=cfg.grad_checkpointing,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=_normalize_lora_target_modules(cfg.lora_target_modules),
        lora_bias=cfg.lora_bias,
    )
    policy_model = loaded_policy.model
    policy_tokenizer = loaded_policy.tokenizer

    reward_handles = [
        _load_reward_handle(adapter_path=path, cfg=cfg, device=device, dtype=dtype)
        for path in reward_adapter_paths
    ]

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=(cfg.betas1, cfg.betas2),
        weight_decay=cfg.weight_decay,
    )
    algo = _build_online_algo(cfg)
    sampler = HFSampler(policy_tokenizer, device=device)
    sampling_cfg = SamplingConfig(
        min_new_tokens=cfg.min_new_tokens,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        do_sample=cfg.temperature > 0.0,
    )

    logger = WandBLogger(
        project=cfg.wandb_project,
        run_name=cfg.wandb_name,
        config=vars(cfg),
        enabled=cfg.wandb_enabled,
        local_dir=output_dir,
    )
    logger.log(
        {
            "setup/trainable_params": float(loaded_policy.trainable_params),
            "setup/total_params": float(loaded_policy.total_params),
            "setup/trainable_fraction": float(loaded_policy.trainable_params / max(1, loaded_policy.total_params)),
            "dataset/train_examples": float(len(train_examples)),
            "dataset/eval_examples": float(len(eval_examples)),
            **{f"dataset/{k}": float(v) for k, v in dataset_info["splits"].items()},
            **get_hardware_metrics(device),
            **get_model_device_metrics(policy_model),
        },
        step=0,
    )

    def run_eval(step: int, phase: str) -> Dict[str, float]:
        metrics, rows, rm_scores = evaluate_policy_with_reward_model(
            policy_model=policy_model,
            policy_tokenizer=policy_tokenizer,
            reward_handles=reward_handles,
            cfg=cfg,
            examples=eval_examples,
            device=device,
            max_prompt_tokens=cfg.max_prompt_tokens,
            max_response_tokens=cfg.max_response_tokens,
            generation_max_new_tokens=cfg.eval_max_new_tokens,
            temperature=cfg.eval_temperature,
            top_p=cfg.eval_top_p,
            generation_batch_size=cfg.eval_batch_size,
        )
        logger.log(metrics, step=step)
        logger.log_table(
            f"samples/eval_{phase}",
            _sample_rows_for_logging(
                eval_examples,
                rows,
                rm_scores,
                sample_log_n=cfg.sample_log_n,
                max_chars=cfg.sample_log_max_chars,
            ),
            step=step,
        )
        return metrics

    print("[eval] running baseline evaluation at step=0")
    run_eval(step=0, phase="baseline")

    start_time = time.time()
    replay: deque[ReplayPreferenceExample] = deque(maxlen=max(1, cfg.replay_capacity))
    for step in range(1, cfg.steps + 1):
        maybe_update_warmup_lr(optimizer, cfg.lr, step - 1, cfg.warmup_steps)
        prompt_batch = _sample_prompt_batch(train_examples, cfg.batch_size, rng)
        rollout = sampler.rollout(
            policy_model=policy_model,
            prompt_messages=[ex.prompt_messages for ex in prompt_batch],
            task_names=["synthetic_instruction_following"] * len(prompt_batch),
            task_metas=[
                {
                    "row_id": ex.row_id,
                    "prompt_text": ex.prompt_text,
                    "reference_response_text": ex.reference_response_text,
                }
                for ex in prompt_batch
            ],
            group_size=cfg.group_size,
            sampling=sampling_cfg,
            max_prompt_tokens=cfg.max_prompt_tokens,
            output_to_cpu=False,
        )

        reward_rows = []
        for i, completion_text in enumerate(rollout.completion_texts):
            meta = rollout.task_metas[i]
            reward_rows.append(
                {
                    "row_id": f"{meta.get('row_id', i)}:{i}",
                    "prompt_messages": rollout.prompt_messages[i],
                    "prompt_text": str(meta.get("prompt_text", "")),
                    "response_text": _normalize_completion_for_reward_scoring(completion_text),
                }
            )
        rewards, reward_ensemble_metrics = _score_rows_with_reward_models(
            reward_handles,
            reward_rows,
            cfg=cfg,
            device=device,
        )
        advantages = _compute_advantages(
            rewards,
            cfg.group_size,
            advantage_mode=cfg.advantage_mode,
            divide_by_std=_algo_divides_advantages_by_std(cfg.algo),
        )
        batch = RolloutBatch(
            input_ids=rollout.input_ids,
            attention_mask=rollout.attention_mask,
            completion_mask=rollout.completion_mask,
            old_logprobs=rollout.old_logprobs,
            ref_logprobs=rollout.ref_logprobs,
            rewards=rewards,
            advantages=advantages,
            task_names=rollout.task_names,
            completion_texts=rollout.completion_texts,
        )
        train_metrics = algo.update(
            policy_model,
            optimizer,
            batch,
            grad_accum_steps=cfg.grad_accum_steps,
        )
        replay_added = 0
        replay_metrics: Dict[str, float] = {}
        if cfg.replay_enabled:
            replay_added = _add_rollout_preferences_to_replay(
                replay,
                rollout,
                rewards,
                min_reward_gap=cfg.replay_min_reward_gap,
            )
            replay_metrics = _replay_preference_update(
                model=policy_model,
                optimizer=optimizer,
                replay=replay,
                cfg=cfg,
                tokenizer=policy_tokenizer,
                device=device,
                rng=rng,
            )
        completion_lengths = batch.completion_mask.sum(dim=1).float()
        log_metrics = {
            "rollout/reward_model_score_mean": float(rewards.mean().item()),
            "rollout/reward_model_score_std": float(rewards.std(unbiased=False).item()),
            "rollout/reward_model_score_min": float(rewards.min().item()),
            "rollout/reward_model_score_max": float(rewards.max().item()),
            "rollout/advantage_mean": float(advantages.mean().item()),
            "rollout/advantage_std": float(advantages.std(unbiased=False).item()),
            "rollout/advantage_mode_rank": float(cfg.advantage_mode == "rank"),
            "rollout/completion_mean_tokens": float(completion_lengths.mean().item()),
            "rollout/completion_max_tokens": float(completion_lengths.max().item()),
            "rollout/count_completions": float(rewards.numel()),
            "replay/added_pairs": float(replay_added),
            "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
            "time/seconds_since_start": float(time.time() - start_time),
            **train_metrics,
            **reward_ensemble_metrics,
            **replay_metrics,
            **get_cuda_memory_metrics(prefix="train"),
        }
        logger.log(log_metrics, step=step)

        should_eval = (step % cfg.eval_interval == 0) or (step == cfg.steps)
        should_save = (step % cfg.save_interval == 0) or (step == cfg.steps)
        if should_eval:
            print(f"[eval] running evaluation at step={step}")
            run_eval(step=step, phase=f"step_{step}")
        if should_save:
            print(f"[checkpoint] saving step={step}")
            save_checkpoint(policy_model, cfg, step=step)

    logger.finish()


if __name__ == "__main__":
    main()
