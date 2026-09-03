# -*- coding: utf-8 -*-
import os
from pathlib import Path

import torch
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEM3ReRanker(object):
    def __init__(self, model_path, max_length=4096):
        model_path = str(model_path).strip()
        if not model_path:
            raise ValueError("Reranker 模型路径不能为空")

        local_path = Path(os.path.expandvars(os.path.expanduser(model_path)))
        local_reference = local_path.is_absolute() or model_path.startswith((".", "~"))
        if local_reference:
            if not local_path.is_dir():
                raise FileNotFoundError(
                    f"Reranker 本地模型目录不存在: {local_path}。"
                    "请设置 RAG_RERANKER_MODEL_PATH，或下载仓库默认模型。"
                )
            model_path = str(local_path.resolve())

        # 加载 rerank 模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            self.model.half()
        self.model.to(self.device)
        self.max_length = max_length


    def rank(self, query, candidate_docs, topk=10):
        # 输入文档对，返回每一对(query, doc)的相关得分，并从大到小排序
        pairs = [(query, doc.page_content) for doc in candidate_docs]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            scores = self.model(**inputs).logits
        scores = scores.detach().cpu().clone().numpy()
        response = [
            doc
            for score, doc in sorted(
                zip(scores, candidate_docs), reverse=True, key=lambda x: x[0]
            )
            ][:topk]
        return response


if __name__ == "__main__":
    bge_reranker_large = "./models/BAAI/bge-reranker-v2-m3/"
    # bce_reranker_base = "../../models/bce-reranker-base-v1"
    bge_rerank = BGEM3ReRanker(bge_reranker_large)
    query = "今天天气怎么样"
    docs = ["你好", "今天天气不错", "今天有雨吗"]
    docs = [Document(page_content=doc, metadata={}) for doc in docs]
    response = bge_rerank.rank(query, docs)
    print(response)
