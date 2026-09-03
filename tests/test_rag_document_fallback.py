# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from src.retriever.document_resolver import resolve_milvus_document


def _load_utils_with_empty_mongo():
    class _Collection:
        @staticmethod
        def find_one(_query):
            return None

    class _MongoConfig:
        @staticmethod
        def get_collection(_name):
            return _Collection()

    fake = types.ModuleType("src.client.mongodb_config")
    fake.MongoConfig = _MongoConfig
    path = Path(__file__).resolve().parents[1] / "src" / "utils.py"
    spec = importlib.util.spec_from_file_location("rag_utils_empty_mongo", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    with patch.dict(sys.modules, {"src.client.mongodb_config": fake}):
        spec.loader.exec_module(module)
    return module


class RagDocumentFallbackTests(unittest.TestCase):
    def test_missing_mongo_document_uses_milvus_text(self):
        hit = {"id": "doc-1", "text": "来自 Milvus 索引的手册片段"}

        resolved = resolve_milvus_document(hit, None)

        self.assertIsNotNone(resolved)
        text, metadata = resolved or ("", {})
        self.assertEqual(text, "来自 Milvus 索引的手册片段")
        self.assertEqual(metadata["unique_id"], "doc-1")
        self.assertEqual(metadata["retrieval_source"], "milvus_index")

    def test_mongo_document_keeps_full_text_and_metadata(self):
        hit = {"id": "doc-1", "text": "截断片段"}
        mongo_document = {
            "page_content": "MongoDB 中的完整手册正文",
            "metadata": {"page": 12},
        }

        resolved = resolve_milvus_document(hit, mongo_document)

        self.assertEqual(
            resolved,
            ("MongoDB 中的完整手册正文", {"page": 12, "unique_id": "doc-1"}),
        )

    def test_missing_text_is_skipped(self):
        self.assertIsNone(resolve_milvus_document({"id": "doc-1"}, None))

    def test_missing_parent_document_keeps_current_chunk(self):
        utils = _load_utils_with_empty_mongo()
        child = Document(
            page_content="仍可用于回答的子片段",
            metadata={"unique_id": "child-1", "parent_id": "missing-parent"},
        )

        merged = utils.merge_docs([child], [])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].page_content, "仍可用于回答的子片段")

    def test_post_processing_allows_missing_image_and_page_metadata(self):
        utils = _load_utils_with_empty_mongo()
        docs = [Document(page_content="手册片段", metadata={"unique_id": "doc-1"})]

        result = utils.post_processing("答案【1】", docs)

        self.assertEqual(result["answer"], "答案")
        self.assertEqual(result["cite_pages"], [])
        self.assertEqual(result["related_images"], [])


if __name__ == "__main__":
    unittest.main()
