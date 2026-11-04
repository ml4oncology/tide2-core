"""Tests for keep_dates: DATE/DATE_TIME entities left unchanged, everything else anonymized."""

import json

import pytest

from tide2.actors.anonymizer import MAX_ANON_CHUNK_SIZE
from tide2.actors.anonymizer import AnonymizerWorker
from tide2.actors.anonymizer import create_anonymizer_actor
from tide2.anonymizers import presidio_patches

SALT = b"\x00" * 32
KEY = b"\x11" * 32
TEXT = "Pt John Smith seen July 11, 2025 at 10:30 at Toronto General, MRN 12345678."
SPANS = [("PATIENT", 3, 13), ("DATE", 19, 32), ("DATE_TIME", 36, 41), ("HOSPITAL", 45, 60), ("MRN", 66, 74)]


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


def test_dates_are_kept_and_other_entities_anonymized():
    out = _process(AnonymizerWorker(salt=SALT, key=KEY, keep_dates=True), TEXT, SPANS)
    text = out["anonymized_note_text"]
    assert out["stage_status"] == "success"
    assert "July 11, 2025" in text
    assert "10:30" in text
    assert "John Smith" not in text
    assert "12345678" not in text
    assert out["jitter"] is None
    operators = {item["entity_type"]: item["operator"] for item in json.loads(out["anonymizer_results_json"])}
    assert operators["DATE"] == "keep"
    assert operators["DATE_TIME"] == "keep"
    assert operators["PATIENT"] != "keep"


def test_dates_are_shifted_without_keep_dates():
    out = _process(AnonymizerWorker(salt=SALT, key=KEY), TEXT, SPANS)
    assert "July 11, 2025" not in out["anonymized_note_text"]
    assert out["jitter"] == 5


def test_missing_jitter_is_not_an_error_with_jitter_required():
    out = _process(AnonymizerWorker(salt=SALT, key=KEY, keep_dates=True, jitter_required=True), TEXT, SPANS, None)
    assert out["stage_status"] == "success"
    assert "July 11, 2025" in out["anonymized_note_text"]
    assert out["jitter"] is None


def test_combines_with_mask_all():
    out = _process(AnonymizerWorker(salt=None, key=None, mask_all=True, keep_dates=True), TEXT, SPANS)
    assert out["anonymized_note_text"] == "Pt [PATIENT] seen July 11, 2025 at 10:30 at [HOSPITAL], MRN [MRN]."
    assert out["jitter"] is None


def test_dates_are_kept_across_chunks_of_a_long_note():
    filler = "x" * MAX_ANON_CHUNK_SIZE
    text = TEXT + filler + TEXT
    offset = len(TEXT) + len(filler)
    spans = SPANS + [(t, s + offset, e + offset) for t, s, e in SPANS]
    out = _process(AnonymizerWorker(salt=None, key=None, mask_all=True, keep_dates=True), text, spans)
    expected = "Pt [PATIENT] seen July 11, 2025 at 10:30 at [HOSPITAL], MRN [MRN]."
    assert out["anonymized_note_text"] == expected + filler + expected


def test_factory_passes_keep_dates():
    actor = create_anonymizer_actor(salt=SALT, key=KEY, keep_dates=True)()
    assert "July 11, 2025" in _process(actor, TEXT, SPANS)["anonymized_note_text"]
