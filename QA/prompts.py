"""QA prompt templates following the FlowGen paper (ICLR 2026, Appendix E).

FlowGen uses a single system prompt across all flowchart QA prompts:

    "You are an expert assistant specialized in flowchart understanding.
     Your task is to parse flowchart images into structured knowledge triplets
     that capture both topology and semantics."

We adopt this for both FlowLearn (Yes/No verifiable statements) and FlowVQA
(open-ended QA). The optional self-extracted-triplet block from FlowGen is
omitted — we evaluate pure end-to-end QA.
"""

# ============================================================
#  Shared system prompt (FlowGen Appendix E)
# ============================================================
SYSTEM_PROMPT = (
    "You are an expert assistant specialized in flowchart understanding. "
    "Your task is to parse flowchart images into structured knowledge triplets "
    "that capture both topology and semantics."
)


# ============================================================
#  Self-extracted triplet block (FlowGen Appendix E verbatim)
# ============================================================
TRIPLET_PREAMBLE = (
    "The following description, extracted in the form of triplets from the image, "
    "is provided for reference and may contain errors."
)


def format_triplet_block(triplets) -> str:
    """`triplets` is a list of dicts {source, end, condition} (from LoRA arrow_triplets.json).
    Output a FlowGen-style triplet block (or empty string if no triplets)."""
    if not triplets:
        return ""
    lines = [TRIPLET_PREAMBLE]
    for t in triplets:
        src = (t.get("source") or "").strip()
        end = (t.get("end") or "").strip()
        cond = t.get("condition")
        if cond is None or str(cond).strip().lower() in ("", "none"):
            cond = "connectedTo"
        cond = str(cond).strip()
        if not src or not end or end.lower() == "pending":
            continue
        lines.append(f"<{src}, {cond}, {end}>")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


# ============================================================
#  FlowLearn — Yes/No statement verification (FlowGen format)
# ============================================================
_FLOWLEARN_TF_INTRO = (
    "Determine whether the following description of the image is correct "
    "(just answer Yes or No):\n"
    '"{statement}"'
)


def _wrap_with_triplets(intro: str, triplet_block: str = "") -> str:
    parts = [intro]
    if triplet_block:
        parts.append(triplet_block)
    parts.append("The answer is:")
    return "\n".join(parts)


def flowlearn_isTrueFalse_AtoB(a: str, b: str, triplet_block: str = "") -> str:
    """Statement form follows FlowLearn's official `dataset_processing.py`."""
    intro = _FLOWLEARN_TF_INTRO.format(
        statement=f"Arrow points from node '{a}' to node '{b}'."
    )
    return _wrap_with_triplets(intro, triplet_block)


def flowlearn_isTrueFalse_betweenAB(a: str, b: str, triplet_block: str = "") -> str:
    intro = _FLOWLEARN_TF_INTRO.format(
        statement=f"Arrow exists between node '{a}' and node '{b}'."
    )
    return _wrap_with_triplets(intro, triplet_block)


# ----- counting (FlowLearn-official phrasing kept; closer to a verifiable
# numerical question than FlowGen's templates explicitly cover) -----
_FLOWLEARN_NUM_TMPL = (
    "The given image contains a simulated flowchart. You should find all "
    "{target} and determine the total number of {target} in the flowchart. "
    "Answer the question with a number."
)


def flowlearn_num_nodes_prompt(triplet_block: str = "") -> str:
    intro = _FLOWLEARN_NUM_TMPL.format(target="text nodes")
    return _wrap_with_triplets(intro, triplet_block)


def flowlearn_num_arrows_prompt(triplet_block: str = "") -> str:
    intro = _FLOWLEARN_NUM_TMPL.format(target="arrows")
    return _wrap_with_triplets(intro, triplet_block)


# ============================================================
#  FlowVQA — open-ended QA (FlowGen-style system prompt + short answer)
# ============================================================
def flowvqa_qa_prompt(question: str, triplet_block: str = "") -> str:
    intro = (
        "Answer the following question about the flowchart. "
        "Give a short, concise answer.\n\n"
        f"Question: {question}"
    )
    return _wrap_with_triplets(intro, triplet_block)
