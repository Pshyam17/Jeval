import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

spacy_available = False
try:
    import spacy
    spacy.load("en_core_web_sm")
    spacy_available = True
except (ImportError, OSError):
    pass

from jeval.memory.fact_index import FactIndex, extract_entities


@pytest.fixture
def fi(tmp_path):
    return FactIndex(tmp_path / "test.db")


# ── spaCy-free DB-layer tests ─────────────────────────────────────────────────

def test_write_and_retrieve_by_seq_id(fi):
    entities = [
        {"text": "JWT_SECRET", "type": "ALL_CAPS",       "value": None},
        {"text": "src/auth.ts", "type": "file_path",     "value": None},
        {"text": "847",         "type": "CARDINAL",       "value": "847"},
    ]
    fi.write_entities(entities, seq_id=1, segment_id=10, session_id="s1")
    result = fi.get_by_seq_id(1, "s1")
    assert len(result) == 3
    texts = {r["entity_text"] for r in result}
    assert "JWT_SECRET" in texts
    assert "src/auth.ts" in texts
    assert "847" in texts


def test_ref_count_increments_on_duplicate_across_segments(fi):
    entities = [{"text": "JWT_SECRET", "type": "ALL_CAPS", "value": None}]
    fi.write_entities(entities, seq_id=1, segment_id=10, session_id="s1")
    fi.write_entities(entities, seq_id=2, segment_id=11, session_id="s1")
    count = fi.get_ref_count("JWT_SECRET", "s1")
    assert count == 2


def test_ref_count_does_not_cross_sessions(fi):
    entities = [{"text": "JWT_SECRET", "type": "ALL_CAPS", "value": None}]
    fi.write_entities(entities, seq_id=1, segment_id=10, session_id="s1")
    fi.write_entities(entities, seq_id=1, segment_id=10, session_id="s2")
    assert fi.get_ref_count("JWT_SECRET", "s1") == 1
    assert fi.get_ref_count("JWT_SECRET", "s2") == 1


def test_get_by_entity_exact_match(fi):
    entities = [
        {"text": "JWT_SECRET", "type": "ALL_CAPS",       "value": None},
        {"text": "step 79",    "type": "step_reference", "value": "79"},
    ]
    fi.write_entities(entities, seq_id=5, segment_id=20, session_id="s1")
    result = fi.get_by_entity("JWT_SECRET", "s1")
    assert len(result) >= 1
    assert all(r["entity_text"] == "JWT_SECRET" for r in result)
    assert not any(r["entity_text"] == "step 79" for r in result)


def test_get_high_reference_threshold(fi):
    entities = [{"text": "migration", "type": "ORG", "value": None}]
    for i in range(4):
        fi.write_entities(entities, seq_id=i, segment_id=i, session_id="s1")
    high = fi.get_high_reference("s1", min_ref_count=3)
    assert any(r["entity_text"] == "migration" for r in high)
    high_strict = fi.get_high_reference("s1", min_ref_count=5)
    assert not any(r["entity_text"] == "migration" for r in high_strict)


def test_get_ref_count_returns_zero_for_missing(fi):
    assert fi.get_ref_count("nonexistent_entity_xyz", "s1") == 0


def test_write_deduplicates_within_segment(fi):
    entities = [
        {"text": "JWT_SECRET", "type": "ALL_CAPS", "value": None},
        {"text": "JWT_SECRET", "type": "ALL_CAPS", "value": None},
    ]
    fi.write_entities(entities, seq_id=1, segment_id=10, session_id="s1")
    result = fi.get_by_seq_id(1, "s1")
    jwt_results = [r for r in result if r["entity_text"] == "JWT_SECRET"]
    assert len(jwt_results) == 1


# ── spaCy-dependent extraction tests ─────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    return FactIndex(tmp_path / "facts.db")


@pytest.mark.skipif(not spacy_available, reason="spaCy en_core_web_sm not installed")
def test_write_and_retrieve_entities(db):
    entities = [{"text": "JWT_SECRET", "type": "error_code", "value": None}]
    db.write_entities(entities, seq_id=1, segment_id=10, session_id="s1")
    rows = db.get_by_entity("JWT_SECRET", "s1")
    assert len(rows) == 1
    assert rows[0]["entity_text"] == "JWT_SECRET"
    assert rows[0]["ref_count"] == 1


@pytest.mark.skipif(not spacy_available, reason="spaCy en_core_web_sm not installed")
def test_ref_count_increments_on_duplicate(db):
    ents = [{"text": "src/auth.ts", "type": "file_path", "value": None}]
    db.write_entities(ents, seq_id=1, segment_id=1, session_id="s1")
    db.write_entities(ents, seq_id=2, segment_id=2, session_id="s1")
    db.write_entities(ents, seq_id=3, segment_id=3, session_id="s1")
    assert db.get_ref_count("src/auth.ts", "s1") == 3


@pytest.mark.skipif(not spacy_available, reason="spaCy en_core_web_sm not installed")
def test_get_high_reference(db):
    ents = [{"text": "roles_table", "type": "ORG", "value": None}]
    for i in range(4):
        db.write_entities(ents, seq_id=i, segment_id=i, session_id="s1")
    high = db.get_high_reference("s1", min_ref_count=3)
    assert any(r["entity_text"] == "roles_table" for r in high)


@pytest.mark.skipif(not spacy_available, reason="spaCy en_core_web_sm not installed")
def test_get_by_seq_id(db):
    ents = [
        {"text": "step 7", "type": "step_reference", "value": None},
        {"text": "500 ERROR", "type": "error_code", "value": None},
    ]
    db.write_entities(ents, seq_id=7, segment_id=7, session_id="s1")
    rows = db.get_by_seq_id(7, "s1")
    texts = [r["entity_text"] for r in rows]
    assert "step 7" in texts
    assert "500 ERROR" in texts


@pytest.mark.skipif(not spacy_available, reason="spaCy en_core_web_sm not installed")
def test_step_reference_extracted():
    entities = extract_entities("step 79 action: deployed to staging")
    types = [e["type"] for e in entities]
    assert "step_reference" in types


@pytest.mark.skipif(not spacy_available, reason="spaCy en_core_web_sm not installed")
def test_file_path_extracted():
    entities = extract_entities("fixed auth in src/middleware/auth.ts line 42")
    texts = [e["text"] for e in entities]
    assert any("src/middleware/auth.ts" in t for t in texts)


@pytest.mark.skipif(not spacy_available, reason="spaCy en_core_web_sm not installed")
def test_no_duplicate_entities_within_segment():
    entities = extract_entities("step 1 action at step 1 again")
    step_refs = [e for e in entities if e["type"] == "step_reference"]
    texts = [e["text"] for e in step_refs]
    assert len(texts) == len(set(t.lower() for t in texts))


# ── coverage-gap tests (no spaCy required) ───────────────────────────────────

@pytest.mark.skipif(spacy_available, reason="spaCy is installed; ImportError path unreachable")
def test_extract_entities_raises_when_spacy_absent():
    """Lines 45-50: ImportError raised when _spacy_available is False."""
    with pytest.raises(ImportError, match="spaCy and en_core_web_sm are required"):
        extract_entities("migration failed at step 79")


def test_extract_entities_with_mocked_spacy():
    """Lines 51-80: extraction pipeline runs correctly with a mocked NLP model."""
    fake_ent_1 = MagicMock()
    fake_ent_1.text = "JWT_SECRET"
    fake_ent_1.label_ = "ORG"

    fake_ent_2 = MagicMock()
    fake_ent_2.text = "847"
    fake_ent_2.label_ = "CARDINAL"

    fake_doc = MagicMock()
    fake_doc.ents = [fake_ent_1, fake_ent_2]

    fake_nlp = MagicMock(return_value=fake_doc)

    with patch("jeval.memory.fact_index._spacy_available", True), \
         patch("jeval.memory.fact_index._nlp", fake_nlp):
        result = extract_entities(
            "JWT_SECRET mismatch fixed, 847 connections rolled back at step 7 in src/auth.ts"
        )

    texts = {e["text"] for e in result}
    # spaCy NER mock entities
    assert "JWT_SECRET" in texts
    assert "847" in texts
    # regex patterns: step_reference and file_path
    assert any(e["type"] == "step_reference" for e in result)
    assert any(e["type"] == "file_path" for e in result)
    # entity types are correctly set
    org_entries = [e for e in result if e["text"] == "JWT_SECRET"]
    assert org_entries[0]["type"] == "ORG"
    cardinal_entries = [e for e in result if e["text"] == "847"]
    assert cardinal_entries[0]["type"] == "CARDINAL"


def test_count_returns_correct_totals(fi):
    """Lines 163-169: count() with and without session_id filter."""
    # empty db
    assert fi.count("s1") == 0

    entities_a = [
        {"text": "JWT_SECRET", "type": "ALL_CAPS",       "value": None},
        {"text": "step 79",    "type": "step_reference", "value": "79"},
    ]
    entities_b = [
        {"text": "migration",  "type": "ORG",            "value": None},
    ]
    fi.write_entities(entities_a, seq_id=1, segment_id=10, session_id="s1")
    fi.write_entities(entities_b, seq_id=2, segment_id=11, session_id="s1")

    assert fi.count("s1") == 3

    # count is session-scoped
    fi.write_entities(entities_a, seq_id=1, segment_id=10, session_id="s2")
    assert fi.count("s1") == 3
    assert fi.count("s2") == 2

    # nonexistent session
    assert fi.count("nonexistent") == 0

    # count without session_id returns total across all sessions
    assert fi.count() >= 5
