"""
evaluator/evaluator.py
──────────────────────
Retrieves RAG outputs from Supabase, evaluates them using Mistral,
and writes the numerical scores back to Supabase Postgres (rag_evaluations).
"""

import os
import json
import uuid
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so we can import utils
sys.path.append(str(Path(__file__).parent.parent))

from mistralai import Mistral
from loguru import logger
from dotenv import load_dotenv

from utils.cloud_storage import CloudStorage

load_dotenv()

SYSTEM_PROMPT = """You are an expert RAG Evaluator.
Given a User Query, Retrieved Context (matched jobs), and the final Generated Analysis, 
return a JSON object with strictly these keys evaluating the quality from 0.0 to 1.0:
{
  "relevance_score": float,  # Were the matched jobs relevant to the query?
  "faithfulness_score": float, # Was the analysis faithfully derived ONLY from the matched jobs without hallucinating outside facts?
  "reasoning": "A short 2-sentence explanation of your scores."
}
IMPORTANT: Output ONLY valid JSON. Keep it strictly inside a JSON block.
"""

class RAGEvaluator:
    def __init__(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set.")
        
        self.client = Mistral(api_key=api_key)
        self.storage = CloudStorage()
        self.model = "mistral-small-latest"

    def evaluate_output(self, rag_output: dict) -> dict:
        query = rag_output.get("query", "")
        # Limit context to avoid giant payloads
        jobs = rag_output.get("matched_jobs", [])
        context = json.dumps(jobs[:3], indent=2) 
        analysis = rag_output.get("analysis", "")
        
        user_prompt = f"Query: {query}\n\nContext: {context}\n\nAnalysis: {analysis}\n\nEvaluate."
        
        # Mistral JSON mode
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
        try:
            scores = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from Mistral evaluator: {raw_json}")
            scores = {"relevance_score": 0.0, "faithfulness_score": 0.0, "reasoning": "Failed to parse JSON response."}
            
        return scores

    def run(self, limit: int = 5):
        logger.info(f"Fetching {limit} recent outputs from Supabase...")
        outputs = self.storage.get_unevaluated_outputs(limit=limit)
        
        if not outputs:
            logger.warning("No outputs found in 'rag_outputs' table.")
            return

        for out in outputs:
            out_id = out["id"]
            logger.info(f"Evaluating output ID: {out_id}")
            
            scores = self.evaluate_output(out)
            
            eval_record = {
                "id": str(uuid.uuid4()),
                "output_id": out_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "relevance_score": float(scores.get("relevance_score", 0.0)),
                "faithfulness_score": float(scores.get("faithfulness_score", 0.0)),
                "reasoning": scores.get("reasoning", "")
            }
            
            try:
                self.storage.insert_evaluation(eval_record)
                logger.success(f"✓ Recorded evaluation for {out_id[:8]}: "
                               f"Rel={eval_record['relevance_score']}, "
                               f"Faith={eval_record['faithfulness_score']}")
            except Exception as e:
                logger.error(f"Failed to record evaluation: {e}. Is the 'rag_evaluations' table created?")

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    evaluator.run()
