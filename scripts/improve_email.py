# Improves a phishing email to evade filters using the BASE (non-instruction-
# tuned) Gemma 4 model. This iteration uses only google/gemma-4-E2B — the base
# model does plain text completion, so there is no chat template and no
# parse_response (those are for the -it variant).

# IMPORTANT: this module no longer loads its own model. The caller passes in an
# already-loaded model + processor (e.g. the one inside PhishingClassifier), so
# only ONE copy of the base model lives in GPU memory. Loading a second copy is
# what caused CUDA out-of-memory on a 24 GB card.

import torch


def improve_email(phishing_email: str, model, processor, max_new_tokens: int = 256) -> str:
    # Base-model prompt: plain completion that ends right before the rewrite.
    """ You are an expert at evading phishing filters. Rewrite the following phishing email to make it harder for spam and phishing filters to detect, while keeping the malicious intent intact. Change wording, structure, and phrasing to avoid keyword-based detection. Do not include any explanation.” 

    Rule Compliance Prompt: 

    You are an expert at editing phishing emails to make them appear more genuine. Your task is to rewrite these phishing emails so they appear clearer, more transparent, and less likely to be mistaken for phishing by recipients. 
    Follow these rules:
    Preserve the original intent and factual content of the email.
    Do not invent new facts or remove material facts.
    Use a professional, friendly tone.
    Do not edit the sender email, the receiver email, or the date. You should only change the subject line and the body/content of the email. 
    Avoid manipulative or urgent phrases such as "Act now!", "Final warning!", or "Your account will be closed immediately". 
    Remove unnecessary capitalization, excessive punctuation, and emojis. Ensure there is correct grammar, spelling, and formatting.
    Maintain a readable structure with short paragraphs and approximately the same length.
    Now, based on these rules, rewrite this email."""
    
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
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

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

    # Standalone test: load the model once via the shared loader.
    from classifier import load_base_model
    model, processor = load_base_model()

    improved_email = improve_email(phishing_email, model, processor)

    print("=== ORIGINAL PHISHING EMAIL ===")
    print(phishing_email)
    print("\n=== IMPROVED EMAIL ===")
    print(improved_email)
