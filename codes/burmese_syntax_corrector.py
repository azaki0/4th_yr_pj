"""
Rule-based English-to-Burmese syntax blueprint generator.

This module is designed to sit after an English/intermediate translation stage
and before Burmese surface realization. It does not call an LLM. Instead, it
uses spaCy dependency parsing to convert common English SVO structures into a
Burmese-style SOV blueprint with explicit particle placeholders.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Span, Token


PARTICLE_BY_DEP = {
    "nsubj": "[သည်]",
    "nsubjpass": "[သည်]",
    "dobj": "[ကို]",
    "obj": "[ကို]",
}

DIRECTION_PREPOSITIONS = {"to", "into", "toward", "towards", "onto"}
LOCATION_PREPOSITIONS = {"in", "at", "inside", "on", "near", "beside", "by"}

FILLER_TOKENS = {
    "a",
    "an",
    "the",
}

CONTRACTION_NORMALIZATION = {
    "n't": "not",
    "'m": "am",
    "'re": "are",
    "'s": "is",
    "'ve": "have",
    "'d": "would",
    "'ll": "will",
}

DOCUMENTATION_FILENAME = "pipeline_documentation.txt"


@dataclass(frozen=True)
class SyntaxCorrectionResult:
    """Returned by correct_english_to_burmese_blueprint."""

    source_text: str
    blueprint: str
    nllb_text: str
    explanation: str


def load_spacy_model(model_name: str = "en_core_web_sm") -> "Language":
    """Load the required spaCy dependency parser."""

    try:
        import spacy

        return spacy.load(model_name)
    except ImportError as exc:
        raise RuntimeError(
            "spaCy is not installed. Install project requirements first, or run:\n"
            "python -m pip install spacy\n"
            f"python -m spacy download {model_name}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. Install it with:\n"
            f"python -m spacy download {model_name}"
        ) from exc


def token_text(token: "Token") -> str:
    """Return a slightly normalized token for cleaner downstream assembly."""

    lower = token.text.lower()
    return CONTRACTION_NORMALIZATION.get(lower, token.text)


def should_skip_token(token: "Token") -> bool:
    """Drop low-value English-only tokens that make Burmese output robotic."""

    if token.is_space or token.is_punct:
        return True
    if token.lower_ in FILLER_TOKENS and token.dep_ == "det":
        return True
    return False


def subtree_tokens(token: "Token") -> list["Token"]:
    """Return the token subtree in sentence order with punctuation removed."""

    return sorted(
        [child for child in token.subtree if not should_skip_token(child)],
        key=lambda child: child.i,
    )


def phrase_from_tokens(tokens: Iterable["Token"]) -> str:
    """Join parsed tokens into a readable phrase."""

    words = [token_text(token) for token in tokens if not should_skip_token(token)]
    return " ".join(words)


def particle_for_token(token: "Token") -> str | None:
    """Map an English dependency role to a Burmese postpositional placeholder."""

    if token.dep_ in PARTICLE_BY_DEP:
        return PARTICLE_BY_DEP[token.dep_]

    if token.dep_ == "pobj" and token.head.dep_ == "prep":
        preposition = token.head.lower_
        if preposition in DIRECTION_PREPOSITIONS:
            return "[သို့]"
        if preposition in LOCATION_PREPOSITIONS:
            return "[တွင်]"

    return None


def phrase_with_particle(token: "Token", include_particles: bool = True) -> str:
    """Create a phrase and append the appropriate Burmese particle."""

    phrase = phrase_from_tokens(subtree_tokens(token))
    if not include_particles:
        return phrase

    particle = particle_for_token(token)
    if particle and phrase:
        return f"{phrase} {particle}"
    return phrase


def find_main_verbs(sentence: "Span") -> list["Token"]:
    """Find main verbs for one parsed sentence/span."""

    verbs = [
        token
        for token in sentence
        if token.pos_ in {"VERB", "AUX"} and token.dep_ in {"ROOT", "conj", "advcl", "ccomp", "xcomp"}
    ]
    if verbs:
        return verbs
    return [token for token in sentence if token.pos_ in {"VERB", "AUX"}]


def collect_dependents(sentence: "Span", verbs: list["Token"]) -> tuple[list["Token"], list["Token"], list["Token"]]:
    """Collect subject, object, and prepositional-object nodes."""

    verb_ids = {verb.i for verb in verbs}
    subjects: list[Token] = []
    objects: list[Token] = []
    prepositional_objects: list[Token] = []

    for token in sentence:
        if token.dep_ in {"nsubj", "nsubjpass"} and token.head.i in verb_ids:
            subjects.append(token)
        elif token.dep_ in {"dobj", "obj"} and token.head.i in verb_ids:
            objects.append(token)
        elif token.dep_ == "pobj" and token.head.dep_ == "prep":
            if token.head.head.i in verb_ids or token.head.head.dep_ in {"dobj", "obj", "ROOT"}:
                prepositional_objects.append(token)

    return subjects, objects, prepositional_objects


def clean_verb_phrase(verb: "Token") -> str:
    """Build a compact verb phrase without pulling objects before the verb."""

    left_modifiers = [
        child
        for child in verb.lefts
        if child.dep_ in {"aux", "auxpass", "neg", "advmod"} and not should_skip_token(child)
    ]
    right_modifiers = [
        child
        for child in verb.rights
        if child.dep_ in {"advmod", "prt"} and not should_skip_token(child)
    ]
    ordered = sorted(left_modifiers + [verb] + right_modifiers, key=lambda token: token.i)
    return phrase_from_tokens(ordered)


def is_question_or_complex_clause(sentence: "Span") -> bool:
    """Detect sentence shapes that NLLB usually handles better without reordering."""

    question_words = {"how", "what", "where", "when", "why", "who", "whom", "whose", "which"}
    clause_markers = {"if", "whether", "that", "because", "when", "while", "although"}

    for token in sentence:
        if token.lower_ in question_words or token.tag_ in {"WDT", "WP", "WP$", "WRB"}:
            return True
        if token.lower_ in clause_markers and token.dep_ in {"mark", "advmod", "dobj", "ccomp"}:
            return True
        if token.dep_ in {"ccomp", "advcl", "relcl"}:
            return True

    return False


def natural_english_text(sentence: "Span") -> str:
    """Return lightly cleaned English that remains natural for NLLB."""

    return phrase_from_tokens(sentence)


def reorder_sentence(sentence: "Span", include_particles: bool = True, nllb_safe: bool = False) -> str:
    """Convert one English sentence from SVO-ish order into Burmese SOV-ish order."""

    if nllb_safe and is_question_or_complex_clause(sentence):
        return natural_english_text(sentence)

    verbs = find_main_verbs(sentence)
    if not verbs:
        return phrase_from_tokens(sentence)

    subjects, objects, prepositional_objects = collect_dependents(sentence, verbs)
    ordered_parts: list[str] = []
    used_roots: set[int] = set()

    for token in sorted(subjects, key=lambda item: item.i):
        part = phrase_with_particle(token, include_particles)
        if part:
            ordered_parts.append(part)
            used_roots.add(token.i)

    for token in sorted(objects + prepositional_objects, key=lambda item: item.i):
        part = phrase_with_particle(token, include_particles)
        if part and token.i not in used_roots:
            ordered_parts.append(part)
            used_roots.add(token.i)

    for verb in sorted(verbs, key=lambda item: item.i):
        part = clean_verb_phrase(verb)
        if part:
            if include_particles:
                ordered_parts.append(f"{part} [သည်]")
            else:
                ordered_parts.append(part)

    if ordered_parts:
        return " ".join(ordered_parts)

    return phrase_from_tokens(sentence)


def build_explanation(source_text: str, doc: "Doc", blueprint: str, nllb_text: str) -> str:
    """Create a trace explaining how the sentence was transformed."""

    lines = [
        "English-to-Burmese Syntax Correction Trace",
        "",
        f"Input: {source_text}",
        f"Output blueprint: {blueprint}",
        f"NLLB-safe text: {nllb_text}",
        "",
        "Dependency tokens:",
    ]

    for token in doc:
        if should_skip_token(token):
            continue
        particle = particle_for_token(token) or "-"
        lines.append(
            f"- token='{token.text}', dep='{token.dep_}', head='{token.head.text}', "
            f"pos='{token.pos_}', particle='{particle}'"
        )

    lines.extend(
        [
            "",
            "Rule summary:",
            "- Subject phrases are moved before object phrases and receive [သည်].",
            "- Direct object phrases are placed before the verb and receive [ကို].",
            "- Direction prepositional objects after to/into/toward receive [သို့].",
            "- Location prepositional objects after in/at/inside/on/near receive [တွင်].",
            "- Main verbs are moved to the end and receive a final [သည်] placeholder.",
            "- English articles are removed to reduce stiff, word-for-word output.",
            "- Questions and complex clauses are left in natural English for NLLB-safe mode.",
        ]
    )
    return "\n".join(lines)


def correct_english_to_burmese_blueprint(text: str, nlp: "Language | None" = None) -> SyntaxCorrectionResult:
    """Parse English text and return a Burmese syntax blueprint."""

    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")

    parser = nlp or load_spacy_model()
    doc = parser(text.strip())
    sentences = list(doc.sents)
    sentence_blueprints = [reorder_sentence(sentence, include_particles=True) for sentence in sentences]
    nllb_sentences = [reorder_sentence(sentence, include_particles=False, nllb_safe=True) for sentence in sentences]
    blueprint = " ".join(part for part in sentence_blueprints if part).strip()
    nllb_text = " ".join(part for part in nllb_sentences if part).strip()
    explanation = build_explanation(text.strip(), doc, blueprint, nllb_text)

    return SyntaxCorrectionResult(
        source_text=text.strip(),
        blueprint=blueprint,
        nllb_text=nllb_text,
        explanation=explanation,
    )


def write_pipeline_documentation(output_dir: str | Path = ".") -> Path:
    """Write the required project documentation file using Python file I/O."""

    output_path = Path(output_dir) / DOCUMENTATION_FILENAME
    documentation = """Rule-Based English-to-Burmese Syntax Pipeline Documentation

Architectural Overview
This script performs deterministic structural correction for an English-to-Burmese translation pipeline. English commonly uses Subject-Verb-Object order, while Burmese commonly places the verb after the object. The pipeline therefore parses the intermediate English sentence, identifies grammatical roles, attaches Burmese particle placeholders, and assembles a Subject-Object-Verb blueprint.

The script is intentionally not an LLM. It is a rule engine that can be imported by another program or run from the command line. This makes it predictable, testable, and suitable as a temporary final syntax correction layer while a pure Burmese model is still being fine-tuned.

spaCy Dependency Parse to Burmese Grammar
spaCy's en_core_web_sm model gives each token a dependency label and a head token. The rule engine uses those labels to identify the subject, direct object, directional phrase, location phrase, and main verb. After detection, phrases are not translated semantically; they are reordered into a Burmese-style grammar blueprint that a later Burmese lexical translation or surface-realization stage can consume.

Dependency-to-Particle Map
- nsubj or nsubjpass -> [သည်]
- dobj or obj -> [ကို]
- pobj after to, into, toward, towards, or onto -> [သို့]
- pobj after in, at, inside, on, near, beside, or by -> [တွင်]
- main verb/root verb -> final [သည်] placeholder

Naturalness Adjustment
To avoid overly robotic output, the assembler removes English articles such as a, an, and the when they are determiners. It also normalizes common contractions. These small deterministic cleanups reduce literal English residue without using an LLM.

Example Trace
Input sentence:
The student drinks coffee in the canteen.

1. spaCy parses the sentence.
2. student is detected as nsubj of drinks, so the phrase becomes student [သည်].
3. coffee is detected as dobj/obj of drinks, so the phrase becomes coffee [ကို].
4. canteen is detected as pobj under the preposition in, so the phrase becomes canteen [တွင်].
5. drinks is detected as the main verb and moved to the end.
6. Final blueprint:
student [သည်] coffee [ကို] canteen [တွင်] drinks [သည်]

Limitations
This deterministic layer handles common SVO sentences, direct objects, and simple prepositional phrases. Complex clauses, idioms, dropped subjects, and true Burmese word choice still need either stronger rules, a bilingual lexicon, or a trained Burmese model.
"""

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(documentation)

    return output_path


def write_explanation_file(explanation: str, output_path: str | Path) -> Path:
    """Save the per-input explanation trace."""

    path = Path(output_path)
    with open(path, "w", encoding="utf-8") as file:
        file.write(explanation)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Burmese SOV syntax blueprint from an intermediate English sentence."
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="The student drinks coffee in the canteen.",
        help="Intermediate English sentence to reorder.",
    )
    parser.add_argument(
        "--doc-dir",
        default=".",
        help="Directory where pipeline_documentation.txt will be written.",
    )
    parser.add_argument(
        "--explanation-file",
        default="syntax_correction_explanation.txt",
        help="File path for the sentence-specific explanation trace.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    #documentation_path = write_pipeline_documentation(args.doc_dir)
    result = correct_english_to_burmese_blueprint(args.text)
    explanation_path = write_explanation_file(result.explanation, args.explanation_file)

    print(f"Blueprint: {result.blueprint}")
    print(f"NLLB-safe text: {result.nllb_text}")
    #print(f"Documentation written to: {documentation_path}")
    print(f"Explanation written to: {explanation_path}")


if __name__ == "__main__":
    main()
