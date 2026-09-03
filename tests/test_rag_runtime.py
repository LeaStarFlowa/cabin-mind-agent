# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.rag.service import _rag_error_hint
from src import constant


def _write_fake_model(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"test")


class RagModelPathTests(unittest.TestCase):
    def test_missing_tuned_checkpoint_falls_back_to_base_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            _write_fake_model(base)
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(constant, "bge_reranker_model_path", str(base)),
                patch.object(
                    constant,
                    "bge_reranker_tuned_model_path",
                    str(root / "missing-checkpoint"),
                ),
            ):
                self.assertEqual(constant.resolve_bge_reranker_model_path(), str(base))

    def test_valid_tuned_checkpoint_takes_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            tuned = root / "checkpoint"
            _write_fake_model(base)
            _write_fake_model(tuned)
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(constant, "bge_reranker_model_path", str(base)),
                patch.object(constant, "bge_reranker_tuned_model_path", str(tuned)),
            ):
                self.assertEqual(constant.resolve_bge_reranker_model_path(), str(tuned))

    def test_explicit_model_setting_overrides_defaults(self):
        with patch.dict(
            os.environ,
            {"RAG_RERANKER_MODEL_PATH": "BAAI/bge-reranker-v2-m3"},
            clear=True,
        ):
            self.assertEqual(
                constant.resolve_bge_reranker_model_path(),
                "BAAI/bge-reranker-v2-m3",
            )


class RagErrorHintTests(unittest.TestCase):
    def test_model_path_error_does_not_claim_milvus_lock(self):
        hint = _rag_error_hint(RuntimeError("Repo id must be in the form 'repo_name'"))
        self.assertIn("Reranker", hint)
        self.assertNotIn("占用", hint)

    def test_milvus_lock_gets_lock_specific_hint(self):
        hint = _rag_error_hint(RuntimeError("local milvus.db lock failed"))
        self.assertIn("其他进程占用", hint)


if __name__ == "__main__":
    unittest.main()
