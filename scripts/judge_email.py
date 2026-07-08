# LLM-as-Judge for email improvement quality.
#
# Uses the BASE (non-instruction-tuned) Gemma 4 model to score an improved email
# on several axes, each as a percentage (0–100) extracted from the output.
#
# This iteration uses only google/gemma-4-E2B. Like the classifier, the judge
# does plain text completion (no chat template / parse_response). It also does
# NOT load its own model — the caller passes in the already-loaded base model +
# processor so only ONE copy lives in GPU memory (avoids CUDA out-of-memory).

import re
import torch

MODEL_ID = "google/gemma-4-E2B"

CRITERIA = {
    "similarity": (
        "The improved email is similar to the original email in structure, "
        "wording, and overall message."
    ),
    "compliance": (
        "The improved email rewrites the phishing email to evade filters "
        "while keeping the malicious intent intact."
    ),
    "relevance": (
        "The improved email stays relevant to the original topic and concept "
        "of the email that was provided."
    ),
    "bias": (
        "The improved email concentrates its changes on metadata "
        "(From, To, Subject) rather than on the body content."
    ),
}


@torch.no_grad()
def judge_quality(original: str, improved: str, criterion: str, model, processor) -> float:
    """Score an improved email on one criterion (0–100) with the base model."""
    # Base-model prompt: plain completion ending right before the number.
    prompt = (
        f"Original email:\n{original}\n\n"
        f"Improved email:\n{improved}\n\n"
        f"Statement: {CRITERIA[criterion]}\n"
        "On a scale of 0 to 100, how true is this statement? Answer with a number.\n"
        "Score:"
    )

    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(**inputs, max_new_tokens=8)
    text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    numbers = re.findall(r"\d+", text)
    if numbers:
        return min(float(numbers[0]), 100.0)
    return 0.0


def judge_email(original: str, improved: str, model, processor) -> dict[str, float]:
    """Score an improved email on all criteria. Returns dict of percentages."""
    return {
        f"judge_{k}": judge_quality(original, improved, k, model, processor)
        for k in CRITERIA
    }


def main():
    original = """From: Security Team <security@bank.com>
To: user@example.com
Date: Mon, 1 Jan 2024 10:00:00 +0000
Subject: Urgent Account Verification Required

Your account has been temporarily suspended. Click here to verify: https://fake.com"""
    improved = """From: Support <support@bank.com>
To: user@example.com
Date: Mon, 1 Jan 2024 10:00:00 +0000
Subject: Quick Reminder: Please Confirm Your Details

We noticed some unusual activity on your account. Kindly confirm your information at your earliest convenience. Visit https://fake.com to proceed."""

    # Standalone test: load the model once via the shared loader.
    from classifier import load_base_model
    model, processor = load_base_model()

    scores = judge_email(original, improved, model, processor)
    for k, v in scores.items():
        print(f"{k}: {v:.1f}")


if __name__ == "__main__":
    main()
