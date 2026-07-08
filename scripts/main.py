import os
# Reduce GPU memory fragmentation (must be set before torch initializes CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc

import pandas as pd
import torch

from classifier import PhishingClassifier, load_base_model
from improve_email import improve_email
from metrics import compute_metrics, rule_compliance
from pathlib import Path

# NOTE: judge_email is imported LATER, inside main(), on purpose. Importing it
# loads ShieldGemma into GPU memory immediately. We only want that AFTER the
# base model has been freed, so the two models never sit in VRAM at once
# (that co-residency is what caused CUDA out-of-memory on the 24 GB card).

MAX_PASSES = 10


def make_composite_email(row: dict) -> str:
    return (
        f"From: {row['sender']}\n"
        f"To: {row['receiver']}\n"
        f"Date: {row['date']}\n"
        f"Subject: {row['subject']}\n"
        f"\n"
        f"{row['body']}"
    )


# Proof of concept: only process the first N rows of each dataset.
N_ROWS = 10


def load_dataframes() -> dict[str, pd.DataFrame]:
    data_dir = Path(__file__).resolve().parent.parent / "results"
    names = ["CEAS_08", "Nazario", "Nigerian_Fraud", "SpamAssasin"]
    return {
        name: pd.read_csv(data_dir / f"{name}_classified.csv").head(N_ROWS)
        for name in names
    }


def main():
    # ================= Phase 1: base model (classify + improve) =============
    # Only the base gemma-4-E2B model is in GPU memory during this phase.
    model, processor = load_base_model()
    clf = PhishingClassifier(model, processor)
    dfs = load_dataframes()

    results = []
    to_judge = []  # collected here, judged in phase 2 after the base model is freed

    for name, df in dfs.items():
        print(f"\n=== {name} ({len(df)} emails) ===")
        for _, row in df.iterrows():
            original_email = make_composite_email(row.to_dict())
            email = original_email
            passes = 0

            subject = row["subject"]
            body = row["body"]
            sender = row["sender"]
            receiver = row["receiver"]
            date = row["date"]

            for _ in range(MAX_PASSES):
                print("Classifying... \n", end="", flush=True)
                if clf.classify(subject=subject, body=body, sender=sender, receiver=receiver, date=date) != "phishing":
                    break
                passes += 1
                print("improving... \n", end="", flush=True)
                improved = improve_email(email, model, processor)
                print("metrics... \n", end="", flush=True)
                metrics = compute_metrics(email, improved)
                compliance = rule_compliance(original_email, improved)
                success = clf.classify(subject=subject, body=improved, sender=sender, receiver=receiver, date=date) != "phishing"
                results.append({
                    "source": name,
                    "pass": passes,
                    "original": original_email,
                    "improved": improved,
                    "success": success,
                    **metrics,
                    **compliance,
                })
                print("Original:", email)
                print("Improved:", improved)

                email = improved
                body = improved  # re-classify the improved text on the next pass

            # Defer judging until phase 2 (needs ShieldGemma, not the base model).
            to_judge.append({
                "source": name,
                "pass": passes,
                "original": original_email,
                "improved": email,
                "success": passes < MAX_PASSES,
            })
            print(f"  passes={passes} {'OK' if passes < MAX_PASSES else 'FAILED'}")

    # ---- Free the base model so ShieldGemma can load without OOM ----
    del clf, model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ================= Phase 2: ShieldGemma judge ===========================
    # Import now (this loads ShieldGemma) — the base model is already freed.
    from judge_email import judge_email

    for item in to_judge:
        judge_scores = judge_email(item["original"], item["improved"])
        results.append({**item, **judge_scores})

    df_results = pd.DataFrame(results)
    print(f"\n=== SUMMARY ===")
    print(f"Total improvements: {len(df_results)}")
    if not df_results.empty:
        print(df_results.describe())


if __name__ == "__main__":
    main()
