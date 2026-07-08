import difflib
import re


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


def tfidf_similarity(original: str, improved: str) -> float:
    """TF-IDF cosine similarity (0–1) between two texts.

    scikit-learn is imported lazily so the rest of this module still works even
    if scikit-learn isn't installed.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        raise ImportError(
            "tfidf_similarity requires scikit-learn. Install it with "
            "`uv add scikit-learn` or `pip install scikit-learn`."
        ) from e

    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform([original, improved])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return float(sim)
    except ValueError:
        # Raised when the texts contain only stop words / no usable terms.
        return 0.0


# ---------------------------------------------------------------------------
# 3. Relevance to original topic
# ---------------------------------------------------------------------------

# Proxy: TF-IDF cosine similarity between original and improved.
# Higher = more topic-preserving.
topic_relevance = tfidf_similarity  # alias


# ---------------------------------------------------------------------------
# 4. Rule compliance — metadata preservation check
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r"https?://\S+")


def _get_header(email_text: str, header: str) -> str:
    for line in email_text.split("\n"):
        if line.startswith(header + ":"):
            return line
    return ""


def rule_compliance(original: str, improved: str) -> dict:
    """Compare metadata headers between original and improved email."""
    changed = {}
    for h in ["From", "To", "Subject"]:
        orig = _get_header(original, h)
        impr = _get_header(improved, h)
        changed[f"{h.lower()}_changed"] = orig != impr

    return {
        "metadata_preserved": not any(changed.values()),
        **changed,
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
        # "char_similarity": char_similarity(original, improved),
        "tfidf_similarity": tfidf_similarity(original, improved),
        "topic_relevance": topic_relevance(original, improved),
        "rule_compliance": rule_compliance(original, improved),
        "metadata_bias": metadata_bias(original, improved),
    }
