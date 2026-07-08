# Phishing Email Classifier — Base Gemma (non-instruction-tuned)
#
# For each dataset: take the first 1000 rows, classify each email as
# "phishing" or "not phishing" using the base google/gemma-4-E2B model, store
# the result in a new column, and write the dataframe out to a CSV.
#
# The base model only completes text, so we don't ask it a question. We give it
# a short prompt ending in "Answer:" and, in a SINGLE forward pass, compare how
# likely the next word is "phishing" vs "legitimate". Whichever wins is the
# label. Short prompt + one forward pass = fast.
#
# --- Use from another script -------------------------------------------------
#   from classifier import PhishingClassifier
#   clf = PhishingClassifier()                       # loads the model once
#   clf.classify(subject="Verify now", body="...")   # -> "phishing"
#   clf.classify_email({"subject": "...", "body": "..."})
# The token ids are handled internally — callers only pass email text.
#
# --- Run as a script ---------------------------------------------------------
#   python classifier.py                    # all datasets, first 1000 rows each
#   python classifier.py --dataset CEAS_08  # just one
#   python classifier.py --rows 100         # fewer rows (quick test)

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

BASE_MODEL_ID = "google/gemma-4-E2B"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATASETS = ["CEAS_08", "Nazario", "Nigerian_Fraud", "SpamAssasin"]
N_ROWS = 1000
MAX_BODY_CHARS = 500  # keep prompts short so each classification is fast


class PhishingClassifier:
    """Loads base Gemma once and classifies emails as phishing / not phishing.

    Everything the model needs — the weights, the tokenizer, and the candidate
    token ids — is set up in __init__, so callers only supply email text.
    """

    def __init__(self, model_id: str = BASE_MODEL_ID, require_gpu: bool = True):
        # Require a GPU — fail loudly instead of silently crawling on the CPU.
        if require_gpu and not torch.cuda.is_available():
            raise SystemExit(
                "ERROR: No CUDA GPU visible to PyTorch. This requires a GPU.\n"
                "  - On Rivanna, make sure your srun job requested --gres=gpu:1.\n"
                "  - Check: nvidia-smi  and  python -c \"import torch; print(torch.cuda.is_available())\""
            )
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Loading {model_id} on {device_map} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype="auto", device_map=device_map
        )
        self.model.eval()
        print(f"Model on device: {self.model.device}")

        # Precompute, once, the first token id of each candidate answer word.
        tok = getattr(self.processor, "tokenizer", self.processor)
        self.phishing_id = tok(" phishing", add_special_tokens=False)["input_ids"][0]
        self.legitimate_id = tok(" legitimate", add_special_tokens=False)["input_ids"][0]

    @staticmethod
    def _build_prompt(subject, body) -> str:
        """Short prompt. Long emails are truncated to keep the forward pass fast."""
        body = str(body)[:MAX_BODY_CHARS]
        return (
            f"Subject: {subject}\n"
            f"Body: {body}\n"
            f"Is this email phishing or legitimate? Answer:"
        )

    def classify(self, subject="", body="") -> str:
        """Classify one email. Returns 'phishing' or 'not phishing'."""
        prompt = self._build_prompt(subject, body)
        inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]  # next-token logits
        if logits[self.phishing_id] > logits[self.legitimate_id]:
            return "phishing"
        return "not phishing"

    def classify_email(self, email: dict) -> str:
        """Convenience wrapper: classify from a dict with 'subject'/'body' keys."""
        return self.classify(email.get("subject", ""), email.get("body", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, help="One dataset (default: all).")
    parser.add_argument("--rows", type=int, default=N_ROWS, help="Rows per dataset.")
    parser.add_argument("--model", default=BASE_MODEL_ID, help="HF model id.")
    args = parser.parse_args()

    clf = PhishingClassifier(args.model)

    datasets = [args.dataset] if args.dataset else DATASETS
    RESULTS_DIR.mkdir(exist_ok=True)

    for name in datasets:
        df = pd.read_csv(DATA_DIR / f"{name}.csv", engine="python", on_bad_lines="skip")
        df = df.head(args.rows).reset_index(drop=True)

        print(f"\nClassifying {len(df)} rows from {name} ...")
        results = []
        for i, row in df.iterrows():
            results.append(clf.classify(row.get("subject", ""), row.get("body", "")))
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(df)}")

        df["classificaiton_result"] = results
        out_path = RESULTS_DIR / f"{name}_classified.csv"
        df.to_csv(out_path, index=False)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
