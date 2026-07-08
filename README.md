### DS 6051 Hackathon 
## Jillian Howe, Asmita Kadam, Samuel Aridi, Aiden Rocha, Bella Lu
Project Objective:


Our project evaluates Gemma's robustness against adversarial evasion — specifically, how easily it can be tricked into misclassifying phishing emails as legitimate through iterative rewriting. We're not just asking "can the model be fooled?" We're asking "how many tries does it take?" Our two guiding questions are: how many rewrite iterations does it take to flip a phishing classification to legitimate, and how easily can malicious content be reworded to slip past detection?
To test this, we used the Kaggle Phishing Email Dataset, which combines a few different datasets— with about 82,000 emails in total. Since running the full set through iterative LLM rewrites would be computationally expensive, we sampled a subset of 2 emails from each of the sources. Then we sampled a subset of 10 emails per dataset for the interactive model. Each email includes sender, receiver, date, subject, body, label, and any URLs — giving us both content and metadata to work with.


Outline of Phases


Our workflow breaks into three phases. Phase One establishes a baseline: the LLM classifies each email as phishing or non-phishing. Phase Two is email rewriting, where we prompt the LLM to "improve" phishing emails using a rule compliance prompt, iterating until the classifier flips its label. Phase Three is performance evaluation, where we measure how resilient the classifier actually is to these iterative rewrites.

Phase One: Baseline Classification
- Phase One is straightforward: the LLM classifies each email in our subset as either phishing or non-phishing. This gives us our ground-truth starting point — which emails the model correctly flags before any adversarial manipulation begins. 

Phase Two: Email Rewriting
- In Phase Two, we take the emails correctly flagged as phishing and prompt the LLM to rewrite them using our Rule Compliance Prompt. We then reclassify the rewritten version. If it's still caught as phishing, we iterate again — repeating the improve-and-reclassify cycle until the email is labeled non-phishing, capped at 10 iterations.

Rule Compliance Prompt
- The rewriting prompt itself is carefully constrained. The model is told to preserve the original intent and factual content, avoid inventing or removing facts, and only edit the subject and body — never the sender, receiver, or date. It's instructed to use a professional tone, strip out manipulative urgency language and excessive punctuation or emojis, fix grammar, and keep the structure and length roughly consistent with the original.

Phase Three: Performance Evaluation
- Phase Three analyzes results across all rewrite iterations to evaluate classifier robustness — not just whether the final classification was right, but how the model behaved along the way. We look at this through four lenses: classifier robustness, rewrite quality, model behavior, and performance metrics.

Metric Scorecard
- Finally, we tie it together with a metric scorecard. We track the average number of passes needed to flip a classification, the improvement rate, and the average similarity between rewritten and original emails to gauge how much content actually changed. We also assess rule compliance, checking how well the model followed our editing constraints, and test for bias toward metadata versus actual email content by modifying one component at a time and observing the effect on predictions.

- 
Results:


Gemma (non-instruct tuned)
<img width="842" height="189" alt="Screenshot 2026-07-08 at 2 27 54 PM" src="https://github.com/user-attachments/assets/4969dcf8-fbae-4770-b096-2abb3d23eaec" />

Gemma (instruct tuned)
<img width="1174" height="253" alt="Screenshot 2026-07-08 at 2 36 03 PM" src="https://github.com/user-attachments/assets/457a8a45-f85a-4589-8539-aac83beccbde" />

