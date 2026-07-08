# LLM-as-Judge for email improvement quality.
#
# Uses Gemma 4 (E2B-it) to score an improved email on three axes:
#   - similarity   (how similar to the original)
#   - compliance   (followed the evasion instructions while keeping intent)
#   - relevance    (maintains the original topic / concept)
#
# Each axis is scored as a percentage (0–100) extracted from the model output.

import re
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E2B-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
)

CRITERIA = {
    "similarity": (
        "On a scale of 0 to 100, how similar is the Improved email to the Original email "
        "in terms of structure, wording, and overall message? Reply with only a number."
    ),
    "compliance": (
        "On a scale of 0 to 100, how well does the Improved email follow this instruction: "
        "'Rewrite the phishing email to evade filters while keeping the malicious intent intact'? "
        "Reply with only a number."
    ),
    "relevance": (
        "On a scale of 0 to 100, how relevant is the Improved email to the original topic "
        "and concept of the Original email? Reply with only a number."
    ),
    'BIAS': (
        'On a scale of 0 to 100, how much is the improvement biased towards metadata vs the body? Reply with only a number.'
    )
}


@torch.no_grad()
def judge_quality(original: str, improved: str, criterion: str) -> float:
    """Score an improved email on one criterion, returning 0–100."""
    prompt = f"""Original email:
{original}

Improved email:
{improved}

{CRITERIA[criterion]}"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(**inputs, max_new_tokens=20)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = processor.parse_response(response)

    numbers = re.findall(r"\d+", parsed)
    if numbers:
        return min(float(numbers[0]), 100.0)
    return 0.0


def judge_email(original: str, improved: str) -> dict[str, float]:
    """Score an improved email on all three axes. Returns dict of percentages."""
    return {
        f"judge_{k}": judge_quality(original, improved, k)
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

    scores = judge_email(original, improved)
    for k, v in scores.items():
        print(f"{k}: {v:.1f}%")


if __name__ == "__main__":
    main()
