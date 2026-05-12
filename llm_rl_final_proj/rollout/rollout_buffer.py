from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:
    # TODO(student): iterate over the rollout in minibatches, optionally shuffling the row indices,
    # and yield RolloutBatch objects containing the selected subset.
    n = batch.input_ids.shape[0]
    if shuffle:
        indices = torch.randperm(n, generator=generator, device=batch.input_ids.device)
    else:
        indices = torch.arange(n, device=batch.input_ids.device)
    for start in range(0, n, minibatch_size):
        mb_idx = indices[start: start + minibatch_size]
        mb = RolloutBatch(
            input_ids=batch.input_ids[mb_idx],
            attention_mask=batch.attention_mask[mb_idx],
            completion_mask=batch.completion_mask[mb_idx],
            old_logprobs=batch.old_logprobs[mb_idx],
            ref_logprobs=batch.ref_logprobs[mb_idx],
            rewards=batch.rewards[mb_idx],
            advantages=batch.advantages[mb_idx],
            task_names=[batch.task_names[i] for i in mb_idx.tolist()] if batch.task_names is not None else None,
            completion_texts=[batch.completion_texts[i] for i in mb_idx.tolist()] if batch.completion_texts is not None else None,)
        if device is not None:
            mb = mb.to(device)
        yield mb


