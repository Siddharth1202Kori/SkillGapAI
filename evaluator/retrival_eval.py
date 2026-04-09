"""
evaluator/retrival_eval.py
─────────────────────────
Evaluates the raw retrieval performance of the ChromaDB vectors.
Operates as an LLM-as-a-judge specifically grading isolated chunks on:
- Precision@K
- Recall@K
- MRR (Mean Reciprocal Rank)
- nDCG (Normalized Discounted Cumulative Gain)
"""

import os
import json
import uuid
import sys
import math
import time
from datetime import datetime, timezone
from pathlib import Path

# Fix relative imports
sys.path.append(str(Path(__file__).parent.parent))

from mistralai import Mistral
from loguru import logger
from dotenv import load_dotenv

from utils.cloud_storage import CloudStorage

load_dotenv()

SYSTEM_PROMPT = """You are an expert Information Retrieval Evaluator.
Given a User Query and a SPECIFIC contextual chunk retrieved from a vector database,
your job is to determine if this text chunk is RELEVANT to solving the user's query.
Return an integer: 1 if it is relevant/useful, or 0 if it is irrelevant/off-topic.
Output ONLY a strict JSON object:
{
  "relevance": int (1 or 0)
}
"""

class RetrievalEvaluator:
    def __init__(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set.")
        
        self.client = Mistral(api_key=api_key)
        self.storage = CloudStorage()
        self.model = "mistral-small-latest"

    def grade_chunk(self, query: str, chunk_content: str) -> int:
        """Uses Mistral to synthetically grade 0 or 1 for absolute precision."""
        user_prompt = f"Query: {query}\n\nRetrieved Chunk:\n{chunk_content}\n\nIs this chunk relevant? (1 or 0)"
        
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            raw_json = response.choices[0].message.content
            score = json.loads(raw_json).get("relevance", 0)
            return int(score)
        except Exception as e:
            logger.error(f"Failed to evaluate chunk: {e}")
            return 0  # Penalize failures

    def evaluate_output(self, rag_output: dict) -> dict:
        query = rag_output.get("query", "")
        jobs = rag_output.get("matched_jobs", [])
        
        if not jobs:
            logger.warning("No matched jobs found in output. Cannot evaluate retrieval.")
            return {"precision_at_k": 0, "recall_at_k": 0, "mrr": 0, "ndcg": 0, "k_value": 0}
            
        k = len(jobs)
        relevances = []
        
        logger.info(f"Grading top {k} retrieved chunks for query: '{query}'...")
        for idx, job in enumerate(jobs):
            # Extract the raw chunk content that was passed to LLM
            # Capped to 1000 chars to save token costs during synthetic evaluation
            chunk_content = str(job)[:1000] 
            rel = self.grade_chunk(query, chunk_content)
            relevances.append(rel)
            
            # Anti-429 Rate Limit pacing for Mistral free-tier
            time.sleep(2)
            
        logger.info(f"Binary Relevance Array: {relevances}")
        
        # 1. Precision@K (What fraction of retrieved items are relevant?)
        precision = sum(relevances) / k
        
        # 2. Recall@K (Proxy mathematically: since database total true positives is unknown, 
        # we calculate proxy against an assumed synthetic population ceiling)
        assumed_db_total_relevant = max(sum(relevances) + 2, 5) 
        recall = sum(relevances) / assumed_db_total_relevant
        
        # 3. MRR (Mean Reciprocal Rank: How deep until the *first* correct hit?)
        mrr = 0.0
        for idx, rel in enumerate(relevances):
            if rel == 1:
                mrr = 1.0 / (idx + 1)
                break
                
        # 4. nDCG (Normalized Discounted Cumulative Gain: Heavily penalizes relevant chunks placed at the bottom)
        dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_relevances))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        
        return {
            "precision_at_k": round(precision, 3),
            "recall_at_k": round(recall, 3),
            "mrr": round(mrr, 3),
            "ndcg": round(ndcg, 3),
            "k_value": k
        }

    def run(self, limit: int = 1):
        """Fetches the latest execution and assesses its retrieval payload explicitly."""
        logger.info(f"Fetching {limit} recent outputs from 'rag_outputs'...")
        outputs = self.storage.get_unevaluated_outputs(limit=limit)
        
        if not outputs:
            logger.warning("No outputs found.")
            return

        for out in outputs:
            out_id = out["id"]
            logger.info(f"Executing deep Retrieval Evaluation on output: {out_id}")
            
            metrics = self.evaluate_output(out)
            
            # Formulating Supabase Record
            eval_record = {
                "id": str(uuid.uuid4()),
                "output_id": out_id,
                "query": out.get("query", ""),
                "k_value": metrics["k_value"],
                "precision_at_k": float(metrics["precision_at_k"]),
                "recall_at_k": float(metrics["recall_at_k"]),
                "mrr": float(metrics["mrr"]),
                "ndcg": float(metrics["ndcg"]),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            try:
                self.storage.insert_retrieval_evaluation(eval_record)
                logger.success(f"✓ Recorded IR metrics for {out_id[:8]}: "
                               f"P@{metrics['k_value']}={metrics['precision_at_k']}, "
                               f"MRR={metrics['mrr']}, nDCG={metrics['ndcg']}")
            except Exception as e:
                logger.error(f"Failed to record retrieval evaluation: {e}")
                logger.warning("ATTENTION: You must create a 'retrieval_evaluations' table in Supabase "
                               "with the correct datatypes (float8 for scores) to accept this payload.")

if __name__ == "__main__":
    evaluator = RetrievalEvaluator()
    evaluator.run()
