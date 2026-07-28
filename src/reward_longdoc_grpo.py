"""EasyR1 reward adapter: thin HTTP client to the persistent reward_service.

EasyR1 calls compute_score(reward_inputs) with [{"response","ground_truth"}, ...].
We forward the whole batch to the reward service (/score), which holds the embedder +
block-emb cache and calls the vLLM vision judge, and return the fine-grained rewards.
Graceful: if the service is unreachable, return zeros so the trainer keeps running.
"""
import json, os, urllib.request
from typing import Any

SERVICE = os.environ.get("REWARD_SERVICE_URL", "http://127.0.0.1:8001").rstrip("/")
EMPTY = {"overall": 0.0, "answer_correctness": 0, "evidence_relevance": 0,
         "evidence_coverage": 0, "n_quotes": 0, "n_matched": 0}


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    items = [{"response": ri["response"], "ground_truth": ri["ground_truth"]} for ri in reward_inputs]
    body = json.dumps({"items": items}).encode()
    try:
        req = urllib.request.Request(SERVICE + "/score", data=body,
                                     headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
        out = resp["rewards"]
        if len(out) != len(reward_inputs):
            raise ValueError(f"len mismatch {len(out)} vs {len(reward_inputs)}")
        return out
    except Exception as ex:
        print(f"[reward_grpo] service error: {repr(ex)[:200]} -> zeros", flush=True)
        return [dict(EMPTY) for _ in reward_inputs]
