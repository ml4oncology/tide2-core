"""Tests for mask_all: every entity replaced with its [<entity_type>] label."""

import json

import pytest

from tide2.actors.anonymizer import MAX_ANON_CHUNK_SIZE
from tide2.actors.anonymizer import AnonymizerWorker
from tide2.actors.anonymizer import create_anonymizer_actor
from tide2.anonymizers import presidio_patches

TEXT = "Pt John Smith seen 03/04/2020 at Toronto General, MRN 12345678."
SPANS = [("PATIENT", 3, 13), ("DATE", 19, 29), ("HOSPITAL", 33, 48), ("MRN", 54, 62)]
MASKED = "Pt [PATIENT] seen [DATE] at [HOSPITAL], MRN [MRN]."


@pytest.fixture(scope="module", autouse=True)
def _restore_presidio_patches():
    """The real workers patch Presidio globally; undo it for the other test modules."""
    yield
    presidio_patches.enable_whitespace_merging()
    presidio_patches.unpatch_conflict_resolution()


def _process(worker: AnonymizerWorker, text: str, spans: list[tuple[str, int, int]], jitter: int | None = 5) -> dict:
    return worker.process_note(
        note_text=text,
        original_text_hash="h" * 64,
        recognizer_results_json=json.dumps(
            [{"entity_type": t, "start": s, "end": e, "score": 1.0} for t, s, e in spans]
        ),
        patient_id="p1",
        jitter=jitter,
    )


def test_every_entity_is_replaced_with_its_label():
    out = _process(AnonymizerWorker(salt=None, key=None, mask_all=True), TEXT, SPANS)
    assert out["stage_status"] == "success"
    assert out["anonymized_note_text"] == MASKED
    items = json.loads(out["anonymizer_results_json"])
    assert {item["operator"] for item in items} == {"masking"}


def test_unknown_entity_types_are_masked_without_a_fallback():
    out = _process(AnonymizerWorker(salt=None, key=None, mask_all=True), "Code ZX-9.", [("CUSTOM_THING", 5, 9)])
    assert out["anonymized_note_text"] == "Code [CUSTOM_THING]."
    assert out["stage_status"] == "success"


def test_dates_are_masked_and_no_jitter_is_reported():
    out = _process(AnonymizerWorker(salt=None, key=None, mask_all=True, jitter_required=True), TEXT, SPANS, None)
    assert out["stage_status"] == "success"
    assert out["jitter"] is None
    assert "[DATE]" in out["anonymized_note_text"]


def test_masking_holds_across_chunks_of_a_long_note():
    filler = "x" * MAX_ANON_CHUNK_SIZE
    text = TEXT + filler + TEXT
    offset = len(TEXT) + len(filler)
    spans = SPANS + [(t, s + offset, e + offset) for t, s, e in SPANS]
    out = _process(AnonymizerWorker(salt=None, key=None, mask_all=True), text, spans)
    assert out["anonymized_note_text"] == MASKED + filler + MASKED


def test_keys_are_still_required_without_mask_all():
    with pytest.raises(ValueError, match="salt and key"):
        AnonymizerWorker(salt=None, key=None)


def test_factory_accepts_no_keys_with_mask_all():
    actor = create_anonymizer_actor(salt=None, key=None, mask_all=True)()
    assert _process(actor, TEXT, SPANS)["anonymized_note_text"] == MASKED
