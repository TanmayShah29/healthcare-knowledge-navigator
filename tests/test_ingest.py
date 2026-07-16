"""Unit tests for src.ingest — section-aware chunking."""

from src.ingest import _chunk_with_sections


class TestChunkWithSections:
    def test_basic_chunking(self):
        text = "## Section 1\nContent here.\n\n## Section 2\nMore content."
        chunks = _chunk_with_sections(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, tuple) for c in chunks)
        assert all(len(c) == 2 for c in chunks)

    def test_preserves_section_names(self):
        text = "## Dosage\n500mg twice daily.\n\n## Side Effects\nNausea."
        chunks = _chunk_with_sections(text)
        sections = [c[0] for c in chunks]
        assert any("Dosage" in s or "dosage" in s.lower() for s in sections)

    def test_empty_text(self):
        chunks = _chunk_with_sections("")
        assert isinstance(chunks, list)

    def test_single_section(self):
        text = "Just one section with content."
        chunks = _chunk_with_sections(text)
        assert len(chunks) >= 1
