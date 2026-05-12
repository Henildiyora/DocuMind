"""Incremental index stats, delete/rename cleanup, embedding settings mismatch."""

from __future__ import annotations

import json
from pathlib import Path

from documind.config import load_config
from documind.index import DocuMindIndex


def test_incremental_embed_only_changed_file(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("AAA_UNIQUE = 1\n" * 50, encoding="utf-8")
    (tmp_path / "b.py").write_text("BBB_UNIQUE = 2\n" * 50, encoding="utf-8")
    cfg = load_config()
    idx = DocuMindIndex(tmp_path, cfg)
    s1 = idx.build_or_update()
    assert s1.new_files == 2
    assert s1.embedded_chunks > 0

    (tmp_path / "a.py").write_text("AAA_UNIQUE = 99\n" * 50, encoding="utf-8")

    s2 = idx.build_or_update()
    assert s2.changed_files == 1
    assert s2.unchanged_files == 1
    assert s2.embedded_chunks > 0
    idx.close()


def test_delete_file_removes_chunks_after_reindex(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("KEEP_UNIQUE = 1\n" * 40, encoding="utf-8")
    (tmp_path / "gone.py").write_text("GONE_UNIQUE = 2\n" * 40, encoding="utf-8")
    cfg = load_config()
    idx = DocuMindIndex(tmp_path, cfg)
    idx.build_or_update()
    (tmp_path / "gone.py").unlink()
    s = idx.build_or_update()
    assert s.removed_files >= 1
    rows = idx.conn.execute(
        "SELECT rel_path FROM chunks WHERE rel_path = ?",
        ("gone.py",),
    ).fetchall()
    assert rows == []
    idx.close()


def test_embedding_dim_mismatch_in_state_triggers_rebuild(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("print('hi')\n" * 30, encoding="utf-8")
    cfg = load_config()
    idx = DocuMindIndex(tmp_path, cfg)
    idx.build_or_update()
    state_path = tmp_path / ".documind" / "state.json"
    data = json.loads(state_path.read_text(encoding="utf-8"))
    data["embedding_dim"] = 1
    state_path.write_text(json.dumps(data), encoding="utf-8")

    s = idx.build_or_update()
    assert s.new_files >= 1
    data2 = json.loads(state_path.read_text(encoding="utf-8"))
    assert int(data2["embedding_dim"]) == int(cfg.embedding_dim)
    idx.close()


def test_scan_files_force_rehash_reads_after_mtime_trick(tmp_path: Path) -> None:
    """With ``force_rehash=True``, always re-read bytes even when mtime/size match."""
    import os

    from documind.chunker import scan_files

    p = tmp_path / "t.py"
    p.write_text("v1\n", encoding="utf-8")
    cfg = load_config()
    rec1 = scan_files(tmp_path, cfg, known_files={}, force_rehash=False)[0]
    st = p.stat()
    p.write_text("v2\n", encoding="utf-8")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))

    fake_known = {rec1.rel_path: (rec1.file_hash, rec1.mtime, rec1.size)}
    rec_stale = scan_files(tmp_path, cfg, known_files=fake_known, force_rehash=False)[0]
    assert rec_stale.file_hash == rec1.file_hash

    rec_ok = scan_files(tmp_path, cfg, known_files=fake_known, force_rehash=True)[0]
    assert rec_ok.file_hash != rec1.file_hash
