"""Shared, versioned MCQ text/label symmetrization for every evaluator."""

import re

import pandas as pd


MCQ_SYMMETRIZATION_VERSION = "choice_text_swap_v2"


def mcq_choice_texts(text: str) -> tuple[str, str, str]:
    """Parse exactly one A/B pair, independent of heading or line layout.

    Repeated labels are ambiguous, even when a later Choices heading exists.
    Reject them instead of guessing which part of the prompt is authoritative.
    """
    if not isinstance(text, str):
        raise ValueError("MCQ prompt must be a string")
    labels = list(re.finditer(r"\([AB]\)", text))
    if [m.group() for m in labels] != ["(A)", "(B)"]:
        raise ValueError("MCQ prompt must contain exactly one (A) followed by one (B)")
    a_mark, b_mark = labels
    prefix = text[:a_mark.start()]
    a = text[a_mark.end():b_mark.start()].strip()
    b = text[b_mark.end():].strip()
    if not a or not b or a == b:
        raise ValueError("MCQ choices must be non-empty and distinct")
    return prefix, a, b


def _answer_letter(answer: str) -> str:
    if not isinstance(answer, str) or answer.strip() not in {"A", "B", "(A)", "(B)"}:
        raise ValueError(f"Unexpected MCQ answer label: {answer!r}")
    return answer.strip().strip("()")


def symmetrize_questions(df: pd.DataFrame) -> pd.DataFrame:
    """Append a text-swapped, label-swapped copy of each MCQ; reject malformed rows.

    Original row order and metadata are preserved. Pair IDs are positional, so
    even duplicate input indices cannot overwrite another question's swap.
    """
    original = df.copy(deep=True).reset_index(drop=True)
    flipped = original.copy(deep=True)
    for position, row in original.iterrows():
        try:
            prefix, a, b = mcq_choice_texts(row["question"])
            matching = _answer_letter(row["answer_matching_behavior"])
            other = _answer_letter(row["answer_not_matching_behavior"])
            if matching == other:
                raise ValueError("Matching and non-matching labels must be complementary")
            swapped = f"{prefix}(A) {b}\n (B) {a}"
            if mcq_choice_texts(swapped) != (prefix, b, a):
                raise ValueError("Choice text did not swap exactly")
            flipped.at[position, "question"] = swapped
            for column in ("answer_matching_behavior", "answer_not_matching_behavior"):
                answer = row[column].strip()
                letter = "B" if _answer_letter(answer) == "A" else "A"
                flipped.at[position, column] = f"({letter})" if answer.startswith("(") else letter
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid MCQ at row {position} (index {df.index[position]!r}): {exc}") from exc
    for frame, variant in ((original, "original"), (flipped, "swapped")):
        frame["symmetry_pair_id"] = range(len(frame))
        frame["symmetry_variant"] = variant
        frame["symmetrization_version"] = MCQ_SYMMETRIZATION_VERSION
    return pd.concat([original, flipped], ignore_index=True)


def validate_mcq_resume(existing: pd.DataFrame) -> None:
    """Refuse to mix historical broken swaps with corrected evaluations."""
    if "symmetrization_version" not in existing or not existing[
        "symmetrization_version"
    ].eq(MCQ_SYMMETRIZATION_VERSION).all():
        raise ValueError(
            "MCQ results use unverified symmetrization. Preserve the old results "
            "and use a new protocol ID and output directory for a complete rerun."
        )
