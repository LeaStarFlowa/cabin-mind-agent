# 路径都以 infer.py 为参考路径
import os
from pathlib import Path

# Windows 路径（当前项目目录）
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"

# 数据路径
pdf_path = base_dir + "data/Tesla_Manual.pdf"
test_doc_path = base_dir + "data/test_docs.txt"
stopwords_path = base_dir + "data/stopwords.txt"
image_save_dir = base_dir + "data/saved_images"
raw_docs_path = base_dir + "data/processed_docs/raw_docs.pkl"
clean_docs_path = base_dir + "data/processed_docs/clean_docs.pkl"
split_docs_path = base_dir + "data/processed_docs/split_docs.pkl"

# 索引路径
bm25_pickle_path = base_dir + "data/saved_index/bm25retriever.pkl"
tfidf_pickle_path = base_dir + "data/saved_index/tfidfretriever.pkl"
milvus_db_path = base_dir + "data/saved_index/milvus.db"
faiss_db_path = base_dir + "data/saved_index/faiss.db"
faiss_qwen_db_path = base_dir + "data/saved_index/faiss_qwen.db"


# 模型路径
m3e_small_model_path = base_dir + "models/AI-ModelScope/m3e-small"
bge_m3_model_path = base_dir + "models/BAAI/bge-m3"
bce_model_path = base_dir + "models/maidalun/bce-embedding-base_v1"
qwen3_embedding_model_path = base_dir + "models/Qwen3-Embedding-0.6B"
qwen3_reranker_model_path = base_dir + "models/Qwen3-Reranker-0.6B"
qwen3_4b_reranker_model_path = base_dir + "models/Qwen3-Reranker-4B"
bge_reranker_model_path = base_dir + "models/BAAI/bge-reranker-v2-m3"
bge_reranker_tuned_model_path = base_dir + "RAG-Retrieval/rag_retrieval/train/reranker/output/bert/runs/checkpoints/checkpoint_0/"


def _is_local_huggingface_model(path: str) -> bool:
    """判断目录是否至少包含 Transformers 加载模型所需的核心文件。"""
    model_dir = Path(path)
    weights = (
        model_dir / "model.safetensors",
        model_dir / "pytorch_model.bin",
    )
    return (model_dir / "config.json").is_file() and any(item.is_file() for item in weights)


def resolve_bge_reranker_model_path() -> str:
    """优先使用有效微调模型，缺失时回退到仓库内基础模型。"""
    configured = os.getenv("RAG_RERANKER_MODEL_PATH", "").strip()
    if configured:
        expanded = os.path.expandvars(os.path.expanduser(configured))
        if os.path.isabs(expanded) or configured.startswith((".", "~")):
            return os.path.abspath(expanded)
        return configured

    if _is_local_huggingface_model(bge_reranker_tuned_model_path):
        return bge_reranker_tuned_model_path
    return bge_reranker_model_path


bge_reranker_runtime_model_path = resolve_bge_reranker_model_path()
bge_reranker_minicpm_path = base_dir + "models/bge-reranker-v2-minicpm-layerwise"
text2vec_model_path = base_dir + "models/text2vec-base-chinese"
qwen3_8b_tune_model_name = "Qwen3.5-4B"
