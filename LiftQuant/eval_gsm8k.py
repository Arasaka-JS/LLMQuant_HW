import os

import torch
from datasets import config as datasets_config
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table


GSM8K_DIRECT_TASK = "gsm8k"
GSM8K_REASONING_TASK = "gsm8k_cot"
GSM8K_NUM_FEWSHOT = 4
GSM8K_LIMIT = None
GSM8K_BATCH_SIZE = 8
GSM8K_CACHE_REQUESTS = False
GSM8K_LOG_SAMPLES = False
GSM8K_GEN_KWARGS = None
GSM8K_CACHE_ENV = "GSM8K_CACHE_DIR"


def infer_reasoning_capability(lm, args):
    has_reasoning = args.gsm8k_reasoning_capability == "yes"
    return has_reasoning, f"set by --gsm8k_reasoning_capability {args.gsm8k_reasoning_capability}"


def resolve_gsm8k_task(has_reasoning):
    return GSM8K_REASONING_TASK if has_reasoning else GSM8K_DIRECT_TASK


def _metric_values(results):
    metrics = {}
    stderr = {}
    for task, result in results.get("results", {}).items():
        task_metrics = {}
        task_stderr = {}
        for key, value in result.items():
            if key.endswith("_stderr") or "_stderr," in key:
                task_stderr[key] = round(value, 4)
            elif key.startswith(("exact_match", "acc", "acc_norm")):
                task_metrics[key] = round(value, 4)
        if task_metrics:
            metrics[task] = task_metrics
        if task_stderr:
            stderr[task] = task_stderr
    return metrics, stderr


def configure_gsm8k_cache(logger):
    cache_dir = os.environ.get(GSM8K_CACHE_ENV)
    if not cache_dir:
        raise RuntimeError(f"{GSM8K_CACHE_ENV} must be set before running GSM8K evaluation")

    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    datasets_config.HF_DATASETS_CACHE = cache_dir
    datasets_config.HF_DATASETS_OFFLINE = True
    logger.info(f"GSM8K dataset cache: {cache_dir}")
    logger.info("GSM8K dataset loading is forced offline")


@torch.no_grad()
def evaluate_gsm8k(lm, args, logger):
    configure_gsm8k_cache(logger)

    has_reasoning, reason = infer_reasoning_capability(lm, args)
    task = resolve_gsm8k_task(has_reasoning)

    logger.info(f"GSM8K reasoning capability: {has_reasoning} ({reason})")
    logger.info(
        "GSM8K config: "
        f"task={task}, fewshot={GSM8K_NUM_FEWSHOT}, limit=full, "
        f"batch_size={GSM8K_BATCH_SIZE}, cache_requests={GSM8K_CACHE_REQUESTS}"
    )

    if lm.tokenizer.pad_token_id is None:
        lm.tokenizer.pad_token_id = lm.tokenizer.eos_token_id
        logger.info("Set tokenizer.pad_token_id to eos_token_id for GSM8K generation")

    with torch.amp.autocast(device_type="cuda", dtype=args.dtype, enabled=torch.cuda.is_available()):
        hflm = HFLM(
            pretrained=lm.model,
            tokenizer=lm.tokenizer,
            batch_size=GSM8K_BATCH_SIZE,
            device=str(lm.device),
            dtype=args.dtype,
        )
        results = evaluator.simple_evaluate(
            hflm,
            tasks=[task],
            num_fewshot=GSM8K_NUM_FEWSHOT,
            limit=GSM8K_LIMIT,
            cache_requests=GSM8K_CACHE_REQUESTS,
            log_samples=GSM8K_LOG_SAMPLES,
            gen_kwargs=GSM8K_GEN_KWARGS,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )

    logger.info(make_table(results))
    metrics, stderr = _metric_values(results)
    if metrics:
        logger.info(f"GSM8K metrics: {metrics}")
    if stderr:
        logger.info(f"GSM8K stderr: {stderr}")

    payload = {
        "status": "ok",
        "task": task,
        "has_reasoning": has_reasoning,
        "reasoning_reason": reason,
        "metrics": metrics,
        "stderr": stderr,
        "results": results,
    }
    return payload
