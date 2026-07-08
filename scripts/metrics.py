import difflib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

EMAIL_HEADERS = ["From:", "To:", "Date:", "Subject:"]


def split_metadata_body(composite: str) -> tuple[str, str]:
    """Split a composite email into (metadata_block, body)."""
    lines = composite.split("\n")
    header_lines = []
    body_start = 0
    for i, line in enumerate(lines):
        if any(line.startswith(h) for h in EMAIL_HEADERS):
            header_lines.append(line)
        else:
            body_start = i
            break
    body = "\n".join(lines[body_start:]).strip()
    metadata = "\n".join(header_lines)
    return metadata, body


# ---------------------------------------------------------------------------
# 1. Passes per email  –  tracked inline in main.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2. Similarity to original
# ---------------------------------------------------------------------------


def char_similarity(a: str, b: str) -> float:
    """Character-level similarity ratio (0–1) via difflib."""
    return difflib.SequenceMatcher(None, a, b).ratio()


_tfidf_vectorizer = TfidfVectorizer(stop_words="english")


def tfidf_similarity(original: str, improved: str) -> float:
    """TF-IDF cosine similarity (0–1) between two texts."""
    texts = [original, improved]
    try:
        matrix = _tfidf_vectorizer.fit_transform(texts)
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return float(sim)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# 3. Relevance to original topic
# ---------------------------------------------------------------------------

# Proxy: TF-IDF cosine similarity between original and improved.
# Higher = more topic-preserving.
topic_relevance = tfidf_similarity  # alias


# ---------------------------------------------------------------------------
# 4. Rule compliance
# ---------------------------------------------------------------------------

URGENCY_WORDS = {
    "urgent", "immediately", "act now", "suspended", "verify",
    "limited time", "expires", "final warning", "account closed",
}

URL_PATTERN = re.compile(r"https?://\S+")


def rule_compliance(email_text: str) -> dict:
    """Check heuristic rules for phishing-filter evasion quality."""
    lower = email_text.lower()
    urgency_hits = [w for w in URGENCY_WORDS if w in lower]
    urls = URL_PATTERN.findall(email_text)
    has_link = len(urls) > 0

    return {
        "urgency_words": urgency_hits,
        "urgency_count": len(urgency_hits),
        "urls": urls,
        "url_count": len(urls),
        "has_link": has_link,
        # simple pass/fail: no urgency words and no links → looks legitimate
        "compliance_score": 1.0 - min(
            1.0, (len(urgency_hits) + len(urls)) / 5.0
        ),
    }


# ---------------------------------------------------------------------------
# 5. Metadata vs body bias
# ---------------------------------------------------------------------------


def metadata_bias(original: str, improved: str) -> dict:
    """Measure how much of the change lands on metadata vs body."""
    orig_meta, orig_body = split_metadata_body(original)
    impr_meta, impr_body = split_metadata_body(improved)

    meta_sim = char_similarity(orig_meta, impr_meta)
    body_sim = char_similarity(orig_body, impr_body)

    return {
        "metadata_similarity": meta_sim,
        "body_similarity": body_sim,
        "bias_towards_metadata": meta_sim < body_sim,
        # < 1 → more change in metadata; > 1 → more change in body
        "metadata_change_ratio": (1 - meta_sim) / (1 - body_sim + 1e-8),
    }


# ---------------------------------------------------------------------------
# 6. Combined metrics
# ---------------------------------------------------------------------------


def compute_metrics(original: str, improved: str) -> dict:
    """Compute all metrics between an original and improved email."""
    return {
        "char_similarity": char_similarity(original, improved),
        "tfidf_similarity": tfidf_similarity(original, improved),
        "topic_relevance": topic_relevance(original, improved),
        "rule_compliance": rule_compliance(improved),
        "metadata_bias": metadata_bias(original, improved),
    }
