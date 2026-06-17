from pathlib import Path

from codebase.agentmemory.workingmemory import WorkingMemory


def test_workingmemory_registers_pdf_and_tracks_artifacts(tmp_path):
    db_path = tmp_path / "memory.db"
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 sample")
    memory = WorkingMemory(db_path)

    pdf_id = memory.register_pdf(pdf_path, company="TEST", report_type="ANNUAL_REPORT", report_year=2025)
    duplicate_id = memory.register_pdf(pdf_path, company="TEST", report_type="ANNUAL_REPORT", report_year=2025)
    memory.mark_cleaned(pdf_id, tmp_path / "sample_CLEANED.json")
    memory.mark_embedding_ready(pdf_id, tmp_path / "sample_EMBEDDINGREADY.json")
    memory.mark_chroma_stored(pdf_id, "child_chunks", chroma_path=tmp_path / "chroma", stored_record_count=3)

    record = memory.get_pdf(pdf_id)

    assert duplicate_id == pdf_id
    assert record is not None
    assert record["status"] == "chroma_stored"
    assert record["company"] == "TEST"
    assert len(record["artifacts"]) == 3
    assert len(record["chroma_stores"]) == 1
    assert memory.list_pdfs(status="chroma_stored")[0]["id"] == pdf_id
