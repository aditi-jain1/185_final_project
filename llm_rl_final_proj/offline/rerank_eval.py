from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

BENCHMARK_GEN_KWARGS = dict(
    max_new_tokens=256,
    temperature=1.0,
    top_p=1.0,
    do_sample=True,
)

def load_policy(base_model, adapter_path: str, device):
    return PeftModel.from_pretrained(base_model, adapter_path).to(device)


def generate_completion(model, tokenizer, prompt_ids, device) -> str:
    input_ids = prompt_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(input_ids, **BENCHMARK_GEN_KWARGS)
    response_ids = out[0][input_ids.shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def score_with_rm(rm_model, tokenizer, prompt: str, completion: str, device) -> float:
    text = prompt + completion
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        logits = rm_model(**inputs).logits
    # Take the first vocab element's logit at the last token position
    score = logits[0, -1, 0].item()
    return float(score)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--rm_adapter", required=True, help="Path to trained RM adapter")
    parser.add_argument("--policy_adapters", nargs="+", required=True,
                        help="One adapter path per candidate policy, in order")
    parser.add_argument("--policy_names", nargs="+", required=True,
                        help="Human-readable names matching --policy_adapters")
    parser.add_argument("--eval_dataset", required=True, help="Path to eval split")
    parser.add_argument("--prompt_col", default="prompt")
    parser.add_argument("--n_prompts", type=int, default=200)
    parser.add_argument("--output_json", default="rerank_results.json")
    args = parser.parse_args()

    assert len(args.policy_adapters) == len(args.policy_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print("Loading reward model")
    base_for_rm = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    rm_model = PeftModel.from_pretrained(base_for_rm, args.rm_adapter).to(device)
    rm_model.eval()

    # Load from JSONL or Hugging Face directory
    if args.eval_dataset.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=args.eval_dataset)
        # If dataset is a DatasetDict, get the main split
        if hasattr(dataset, 'keys'):
            dataset = dataset[list(dataset.keys())[0]]
    else:
        dataset = load_from_disk(args.eval_dataset)
    
    print(f"Dataset columns: {dataset.column_names}")
    prompts = [dataset[i][args.prompt_col] for i in range(min(args.n_prompts, len(dataset)))]

    all_completions = {}
    all_rm_scores  = {}

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)

    for name, adapter_path in zip(args.policy_names, args.policy_adapters):
        print(f"Generating with policy: {name}")
        policy = PeftModel.from_pretrained(base_model, adapter_path).to(device)
        policy.eval()
        completions, scores = [], []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            completion = generate_completion(policy, tokenizer, inputs["input_ids"][0], device)
            score = score_with_rm(rm_model, tokenizer, prompt, completion, device)
            completions.append(completion)
            scores.append(score)
        all_completions[name] = completions
        all_rm_scores[name]   = scores
        policy.cpu()

    rerank_scores = []
    rerank_chosen = []
    for i in range(len(prompts)):
        best_name = max(args.policy_names, key=lambda n: all_rm_scores[n][i])
        rerank_scores.append(all_rm_scores[best_name][i])
        rerank_chosen.append(best_name)

    results = {
        "per_policy_mean_rm_score": {
            name: sum(all_rm_scores[name]) / len(all_rm_scores[name])
            for name in args.policy_names
        },
        "rerank_mean_rm_score": sum(rerank_scores) / len(rerank_scores),
        "rerank_selection_counts": {
            name: rerank_chosen.count(name) for name in args.policy_names
        },
    }

    print(json.dumps(results, indent=2))
    Path(args.output_json).write_text(json.dumps(results, indent=2))
    print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()