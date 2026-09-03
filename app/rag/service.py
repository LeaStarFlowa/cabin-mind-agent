# -*- coding: utf-8 -*-
"""RAG 薄封装：复用现有 context.rag_engine.RAGEngine（策略保持原样）。"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

_RETRY_AFTER_SEC = 12.0
_MAX_IMAGES_PER_DOC = 8


def _rag_error_hint(error: Exception) -> str:
    """根据实际异常给出针对性提示，避免把所有加载错误都归因于 Milvus。"""
    message = str(error).lower()
    if "repo id must be in the form" in message or "reranker 本地模型目录不存在" in message:
        return (
            "[RAG] Reranker 模型路径无效。请检查 RAG_RERANKER_MODEL_PATH，"
            "并确认 IDE 的工作目录是当前仓库。"
        )
    if (
        "opened by another" in message
        or "open local milvus" in message
        or ("milvus.db" in message and "lock" in message)
    ):
        return "[RAG] milvus.db 正被其他进程占用，请只保留一个后端进程。"
    if "cuda" in message:
        return "[RAG] CUDA 初始化失败，请检查显卡环境或切换到 CPU。"
    return "[RAG] 请根据上方异常检查模型、索引和数据库配置。"


def _resolve_image_path(path: str) -> str:
    try:
        from src.utils import convert_db_path_to_local

        return convert_db_path_to_local(path) or path
    except Exception:
        return path


def _card_images(meta: Dict[str, Any]) -> List[Dict[str, str]]:
    """从手册片段 metadata 抽出可给前端展示的插图。"""
    raw = meta.get("images_info") or meta.get("images") or []
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    seen = set()
    for img in raw:
        if not isinstance(img, dict):
            continue
        path = img.get("image_path") or img.get("relative_path") or ""
        if not path:
            continue
        rel = _resolve_image_path(str(path))
        if rel in seen:
            continue
        seen.add(rel)
        title = str(img.get("title") or "").strip()
        out.append({"title": title, "image_path": rel})
        if len(out) >= _MAX_IMAGES_PER_DOC:
            break
    return out


class RagService:
    def __init__(self):
        self._engine = None
        self._error: Optional[str] = None
        self._error_ts: float = 0.0
        self._lock = threading.Lock()
        self._warmed = False

    def _ensure(self):
        if self._engine is not None:
            return
        if self._error and (time.time() - self._error_ts) < _RETRY_AFTER_SEC:
            return
        with self._lock:
            if self._engine is not None:
                return
            if self._error and (time.time() - self._error_ts) < _RETRY_AFTER_SEC:
                return
            try:
                print("[RAG] 正在加载知识库引擎(BM25+Milvus+Reranker)，请稍候...")
                from context.rag_engine import RAGEngine

                self._engine = RAGEngine()
                self._error = None
                self._error_ts = 0.0
                print("[RAG] 引擎加载完成")
            except Exception as e:
                self._error = str(e)
                self._error_ts = time.time()
                self._engine = None
                print(f"[RAG] 引擎加载失败: {e}")
                print(_rag_error_hint(e))

    @property
    def available(self) -> bool:
        self._ensure()
        return self._engine is not None

    def warmup(self) -> bool:
        """启动时预热，避免首问卡在加载模型。"""
        self._ensure()
        if not self._engine:
            return False
        if self._warmed:
            return True
        try:
            print("[RAG] 热身检索中...")
            self._engine.milvus.retrieve_topk("warmup query", topk=3)
            # 顺带热一下 BM25/jieba
            try:
                self._engine.bm25.retrieve_topk("warmup query", topk=3)
            except Exception:
                pass
            self._warmed = True
            print("[RAG] 热身完成，知识问答可直接使用")
            return True
        except Exception as e:
            print(f"[RAG] 热身跳过: {e}")
            self._warmed = True  # 避免反复卡死
            return False

    def retrieve(self, query: str, topk: int = 5) -> List:
        self._ensure()
        if not self._engine:
            raise RuntimeError(self._error or "RAGEngine 不可用")
        return self._engine.retrieve(query, topk=topk)

    def build_context(self, docs: List) -> Tuple[str, List[str]]:
        self._ensure()
        return self._engine.build_context(docs)

    def build_context_cards(self, docs: List) -> Tuple[str, List[Dict[str, Any]]]:
        """结构化检索卡片，供前端展开阅读。"""
        self._ensure()
        context_str, _ = self._engine.build_context(docs)
        cards: List[Dict[str, Any]] = []
        for idx, doc in enumerate(docs):
            meta = getattr(doc, "metadata", None) or {}
            if not isinstance(meta, dict):
                meta = {}
            content = (getattr(doc, "page_content", None) or "").strip()
            page = meta.get("page_number") or meta.get("page") or meta.get("page_id")
            title = meta.get("title") or meta.get("section") or meta.get("header") or f"手册片段 {idx + 1}"
            preview = " ".join(content.split())[:140]
            cards.append(
                {
                    "index": idx + 1,
                    "title": str(title),
                    "page": page,
                    "content": content,
                    "preview": preview + ("…" if len(preview) >= 140 else ""),
                    "images": _card_images(meta),
                }
            )
        return context_str, cards

    def post_process(self, response: str, docs: List) -> Dict[str, Any]:
        self._ensure()
        return self._engine.post_process(response, docs)


_RAG: Optional[RagService] = None


def get_rag_service() -> RagService:
    global _RAG
    if _RAG is None:
        _RAG = RagService()
    return _RAG
