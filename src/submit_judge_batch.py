#!/usr/bin/env python3
"""Submit the judge JSONL to the Gemini Batch API (50% cost, async) and download results.
API key is read from ~/.config/gemini/api_key (chmod 600, never hard-coded/logged). Model is
gemini-3.5-flash by default. Usage:
  create:  python src/submit_judge_batch.py --jsonl outputs/judge/batch_requests.jsonl
  resume:  python src/submit_judge_batch.py --resume <job_name>   (poll+download an existing job)
The batch job name is saved to outputs/judge/job_name.txt so it can be resumed later.
"""
import argparse, os, time, json, sys

def load_key():
    p = os.path.expanduser("~/.config/gemini/api_key")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        sys.exit("ERROR: ~/.config/gemini/api_key is missing or empty. Put your Gemini API key there "
                 "(chmod 600) first: nano ~/.config/gemini/api_key")
    return open(p).read().strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="outputs/judge/batch_requests.jsonl")
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--out", default="outputs/judge/batch_results.jsonl")
    ap.add_argument("--resume", default=None, help="existing batch job name to poll+download")
    ap.add_argument("--poll_sec", type=int, default=60)
    args = ap.parse_args()
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=load_key())

    if args.resume:
        job_name = args.resume
    else:
        up = client.files.upload(file=args.jsonl,
                                 config=types.UploadFileConfig(display_name="citevqa_judge", mime_type="jsonl"))
        job = client.batches.create(model=args.model, src=up.name,
                                    config={"display_name": "citevqa_judge"})
        job_name = job.name
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open("outputs/judge/job_name.txt", "w").write(job_name)
        n = sum(1 for _ in open(args.jsonl))
        print(f"created batch job {job_name} ({n} requests, model={args.model}). saved to job_name.txt")

    done = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    while True:
        job = client.batches.get(name=job_name)
        st = job.state.name
        if st in done:
            break
        print(f"  state={st} ... (batch async, up to 24h)", flush=True)
        time.sleep(args.poll_sec)
    print(f"job finished: {job.state.name}")
    if job.state.name == "JOB_STATE_SUCCEEDED":
        content = client.files.download(file=job.dest.file_name)
        open(args.out, "wb").write(content)
        print(f"downloaded results -> {args.out} ({sum(1 for _ in open(args.out))} lines)")
    else:
        print("job did not succeed; inspect via client.batches.get")

if __name__ == "__main__":
    main()
