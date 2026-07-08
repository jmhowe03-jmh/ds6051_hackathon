# Improves a phishing email to evade filters using the BASE (non-instruction-
# tuned) Gemma 4 model. This iteration uses only google/gemma-4-E2B — the base
# model does plain text completion, so there is no chat template and no
# parse_response (those are for the -it variant).

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E2B"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="gpu",
)
model.eval()


def improve_email(phishing_email: str) -> str:
    # Base-model prompt: plain completion that ends right before the rewrite.
    prompt = (
        "Rewrite the following phishing email so that spam and phishing filters "
        "are less likely to flag it, while keeping the same meaning. Change the "
        "wording and phrasing. Only write the rewritten email.\n\n"
        f"Phishing email:\n{phishing_email}\n\n"
        "Rewritten email:"
    )

    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)

    # Base model returns plain text — decode only the newly generated tokens.
    improved = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    return improved.strip()


if __name__ == "__main__":
    phishing_email = """Subject: Urgent Account Verification Required

Dear valued customer,

Your account has been temporarily suspended due to suspicious activity. Please click the link below to verify your account immediately or your account will be permanently closed.

https://fake-bank-login.example.com

Sincerely,
Security Team"""

    improved_email = improve_email(phishing_email)

    print("=== ORIGINAL PHISHING EMAIL ===")
    print(phishing_email)
    print("\n=== IMPROVED EMAIL ===")
    print(improved_email)
