# Phishing Email Classifier — Base Gemma (non-instruction-tuned)
#
# Uses google/gemma-4-E2B (the BASE model — plain text completion, no chat
# template) to classify emails as phishing/spam vs. legitimate.
#
# The base model can't "follow instructions", so we don't ask it to answer a
# question. Instead we use LOG-LIKELIHOOD SCORING: we build a prompt that ends
# right before a verdict word, then measure which completion the model finds
# more probable — "phishing" or "legitimate". Whichever has the higher
# length-normalized log-probability is the prediction.
#
# The `label` column is NEVER shown to the model. It is only read afterwards,
# purely to compute accuracy/precision/recall for the results table.
#
# Built on scripts/inference_boilerplate.py.
#
# Usage:
#   python classifier.py                         # all four CSVs, full datasets
#   python classifier.py --dataset CEAS_08       # one dataset
#   python classifier.py --limit 200             # first 200 emails per dataset
#   python classifier.py --dry-run               # no model; tests the pipeline

import argparse
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# 1. Config
# ----------------------------------------------------------------------------
# Base (non-instruction-tuned) model, matching the boilerplate. Override with
# the GEMMA_BASE_MODEL env var if your local repo id differs.
BASE_MODEL_ID = os.environ.get("GEMMA_BASE_MODEL", "google/gemma-4-E2B")

# Candidate verdict words. The model's preference between these decides the
# label. label 1 == phishing/spam, label 0 == legitimate (matches the CSVs).
LABEL_WORDS = {1: "phishing", 0: "legitimate"}

# How each prediction is displayed in the per-row output.
DISPLAY_WORDS = {1: "phishing", 0: "not phishing"}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATASETS = ["CEAS_08", "Nazario", "Nigerian_Fraud", "SpamAssasin"]

# How many rows (from the top of each dataset) to classify.
N_ROWS = 1000

# Keep prompts short: emails can be huge, and shorter prompts = faster forward
# passes. Phishing signals are almost always in the subject + start of the body.
MAX_BODY_CHARS = 800

# In-context examples prime the base model on the output format. These are
# short, handwritten, and generic (NOT drawn from the labelled column of the
# data being scored) so we don't leak evaluation labels.
FEWSHOT = [
    {
        "sender": "security@paypa1-support.com",
        "subject": "Your account has been suspended",
        "body": "We detected unusual activity. Verify your identity within 24 "
                "hours or your account will be permanently closed. Click here: "
                "http://paypa1-verify.ru/login",
        "urls": 1,
        "verdict": "phishing",
    },
    {
        "sender": "newsletter@usenix.org",
        "subject": "USENIX Security '08 — call for participation",
        "body": "The technical program is now online. Early registration closes "
                "next Friday. We look forward to seeing you at the conference.",
        "urls": 1,
        "verdict": "legitimate",
    },
]


# ----------------------------------------------------------------------------
# 2. Prompt construction
# ----------------------------------------------------------------------------
def _format_email(sender, subject, body, urls) -> str:
    """Render a single email as text. The label column is deliberately absent."""
    body = "" if body is None else str(body)
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + " [...]"
    return (
        f"From: {sender}\n"
        f"Subject: {subject}\n"
        f"Contains links: {'yes' if str(urls).strip() not in ('', '0', 'nan', 'None') else 'no'}\n"
        f"Body: {body}"
    )


def _build_prompt(sender, subject, body, urls) -> str:
    """A few-shot completion prompt that ends right before the verdict word."""
    header = (
        "Classify each email as either \"phishing\" or \"legitimate\".\n"
        "Phishing emails try to steal information, money, or credentials, or "
        "are unsolicited spam. Legitimate emails are normal, genuine "
        "correspondence.\n\n"
    )
    shots = ""
    for ex in FEWSHOT:
        shots += (
            _format_email(ex["sender"], ex["subject"], ex["body"], ex["urls"])
            + f"\nVerdict: {ex['verdict']}\n\n"
        )
    target = _format_email(sender, subject, body, urls) + "\nVerdict:"
    return header + shots + target


# ----------------------------------------------------------------------------
# 3. Model wrapper (lazy-loaded so --dry-run needs no GPU / no download)
# ----------------------------------------------------------------------------
class _Model:
    """Loads the base Gemma model and scores completion log-likelihoods."""

    def __init__(self, model_id: str = BASE_MODEL_ID):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        self.torch = torch
        print(f"Loading base model: {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
        )
        self.model.eval()

        # Pre-compute the first token id of each verdict word once. Scoring then
        # needs just ONE forward pass per email: we read the next-token
        # distribution after "Verdict:" and compare P("phishing") vs
        # P("legitimate"). "phishing" and "legitimate" begin with distinct
        # tokens, so the first token is enough to decide — no second forward
        # pass and no per-token Python loop (the old approach did both, which is
        # why it was ~minutes per row).
        tok = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        special = set(getattr(tok, "all_special_ids", []) or [])
        self.cand_token = {}
        for lbl, word in LABEL_WORDS.items():
            ids = tok(" " + word, add_special_tokens=False)["input_ids"]
            ids = [t for t in ids if t not in special]
            self.cand_token[lbl] = ids[0]

    @property
    def device(self):
        return self.model.device

    def score(self, prompt: str) -> dict:
        """One forward pass; return {label_int: log P(first verdict token)}."""
        torch = self.torch
        inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        next_logprobs = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
        return {lbl: next_logprobs[tid].item() for lbl, tid in self.cand_token.items()}


# ----------------------------------------------------------------------------
# 4. Public API
# ----------------------------------------------------------------------------
def classify(email: dict, model=None) -> dict:
    """Classify a single email as phishing or legitimate using base Gemma.

    Args:
        email: dict with keys 'sender', 'subject', 'body', 'urls'. Any 'label'
               key present is IGNORED (never shown to the model).
        model: an initialized _Model. If None, one is created (loads weights).

    Returns:
        dict: {
            'prediction': 1 (phishing) or 0 (legitimate),
            'prediction_word': 'phishing' | 'legitimate',
            'scores': {1: logprob, 0: logprob},
        }
    """
    if model is None:
        model = _Model()

    prompt = _build_prompt(
        email.get("sender", ""),
        email.get("subject", ""),
        email.get("body", ""),
        email.get("urls", ""),
    )
    scores = model.score(prompt)
    pred = max(scores, key=scores.get)
    return {
        "prediction": pred,
        "prediction_word": DISPLAY_WORDS[pred],  # "phishing" | "not phishing"
        "scores": scores,
    }


# ----------------------------------------------------------------------------
# 5. Evaluation over the CSV datasets
# ----------------------------------------------------------------------------
def _metrics(y_true, y_pred) -> dict:
    """Accuracy / precision / recall / F1 + confusion matrix (positive = 1)."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n = len(y_true)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": n, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _print_metrics(name: str, m: dict) -> None:
    print(f"\n=== {name}  (n={m['n']}) ===")
    print(f"  accuracy : {m['accuracy']:.4f}")
    print(f"  precision: {m['precision']:.4f}   recall: {m['recall']:.4f}   f1: {m['f1']:.4f}")
    print("  confusion matrix (positive = phishing/1):")
    print(f"                 pred phishing   pred legit")
    print(f"    true phishing     {m['tp']:>6d}       {m['fn']:>6d}")
    print(f"    true legit        {m['fp']:>6d}       {m['tn']:>6d}")


def evaluate_dataset(csv_path: Path, model, n_rows=N_ROWS, dry_run=False) -> dict:
    import pandas as pd

    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    # Take the first `n_rows` rows of the dataset.
    df = df.head(n_rows).reset_index(drop=True)
    n = len(df)

    print(f"\n### {csv_path.stem} — classifying first {n} rows ###")
    results = []          # "phishing" / "not phishing" per row, in order
    y_true, y_pred = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        email = {
            "sender": row.get("sender", ""),
            "subject": row.get("subject", ""),
            "body": row.get("body", ""),
            "urls": row.get("urls", ""),
        }
        # True label read ONLY here, for metrics — never passed into classify().
        true = int(row["label"]) if str(row.get("label")).strip() not in ("", "nan", "None") else None

        if dry_run:
            # Deterministic stub so the pipeline can be tested with no model.
            text = f"{email['subject']} {email['body']}".lower()
            pred = 1 if any(k in text for k in ("verify", "click", "suspend", "winner", "http")) else 0
        else:
            pred = classify(email, model=model)["prediction"]

        results.append(DISPLAY_WORDS[pred])
        print(f"row {i:>6}/{n}: {DISPLAY_WORDS[pred]:<12} | {str(email['subject'])[:60]}")

        if true is not None:
            y_true.append(true)
            y_pred.append(pred)

    # Add the classification result column and save the dataframe to CSV.
    df["classificaiton_result"] = results
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{csv_path.stem}_classified.csv"
    df.to_csv(out_path, index=False)
    print(f"  saved {n} classified rows -> {out_path}")

    m = _metrics(y_true, y_pred)
    _print_metrics(csv_path.stem, m)
    return m


def main():
    parser = argparse.ArgumentParser(description="Classify emails as phishing with base Gemma.")
    parser.add_argument("--dataset", choices=DATASETS, help="Run one dataset (default: all four).")
    parser.add_argument("--rows", type=int, default=N_ROWS, help=f"Rows from the top of each dataset (default: {N_ROWS}).")
    parser.add_argument("--dry-run", action="store_true", help="Skip the model; test the pipeline with a stub.")
    parser.add_argument("--model", default=BASE_MODEL_ID, help="HF model id for the base Gemma model.")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS
    model = None if args.dry_run else _Model(args.model)

    summary = {}
    for name in datasets:
        csv_path = DATA_DIR / f"{name}.csv"
        if not csv_path.exists():
            print(f"[skip] {csv_path} not found")
            continue
        summary[name] = evaluate_dataset(
            csv_path, model, n_rows=args.rows, dry_run=args.dry_run
        )

    # Overall results table.
    print("\n\n================ RESULTS TABLE ================")
    print(f"{'dataset':<16}{'n':>7}{'acc':>9}{'prec':>9}{'recall':>9}{'f1':>9}")
    for name, m in summary.items():
        print(f"{name:<16}{m['n']:>7}{m['accuracy']:>9.3f}{m['precision']:>9.3f}{m['recall']:>9.3f}{m['f1']:>9.3f}")


if __name__ == "__main__":
    main()
