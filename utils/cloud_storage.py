"""
utils/cloud_storage.py
──────────────────────
Handles upload/download of scraped job JSON using Supabase Storage.

Supabase Storage is an S3-compatible object store with a nice dashboard.
Each scrape run is saved as a JSON file in the "indeed-jobs" bucket.

Setup:
  1. Go to supabase.com → your project → Storage
  2. Create a bucket called "indeed-jobs" (set to private)
  3. Copy your project URL + service_role key into .env
"""

import json
import os

from supabase import create_client, Client
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class CloudStorage:
    BUCKET = "indeed-jobs"          # change if you named your bucket differently

    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")   # use service_role key, not anon key

        if not url or not key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env\n"
                "Find them at: supabase.com → project → Settings → API"
            )

        self.client: Client = create_client(url, key)
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Create the storage bucket if it doesn't exist yet."""
        existing = [b.name for b in self.client.storage.list_buckets()]
        if self.BUCKET not in existing:
            self.client.storage.create_bucket(self.BUCKET, options={"public": False})
            logger.info(f"Created Supabase bucket: '{self.BUCKET}'")
        else:
            logger.debug(f"Bucket '{self.BUCKET}' already exists.")

    # ── Upload ─────────────────────────────────────────────────────────────────

    def upload_jobs(self, listings: list[dict], key: str) -> str:
        """
        Upload job listings as JSON to Supabase Storage.

        Args:
            listings: list of job dicts (use dataclasses.asdict() first)
            key:      file path inside bucket, e.g. "jobs/2024-01-15_ml_engineer.json"

        Returns:
            Supabase storage path: supabase://<bucket>/<key>
        """
        body = json.dumps(listings, indent=2, ensure_ascii=False).encode("utf-8")

        # upsert=True → overwrites if file already exists (safe for reruns)
        self.client.storage.from_(self.BUCKET).upload(
            path=key,
            file=body,
            file_options={"content-type": "application/json", "upsert": "true"},
        )

        uri = f"supabase://{self.BUCKET}/{key}"
        logger.success(f"Uploaded {len(listings)} jobs → {uri}")
        return uri

    # ── Download ───────────────────────────────────────────────────────────────

    def download_jobs(self, key: str) -> list[dict]:
        """
        Download and parse a job JSON file from Supabase Storage.

        Args:
            key: file path inside bucket, e.g. "jobs/2024-01-15_ml_engineer.json"

        Returns:
            List of job dicts
        """
        raw: bytes = self.client.storage.from_(self.BUCKET).download(key)
        data = json.loads(raw.decode("utf-8"))
        logger.info(f"Downloaded {len(data)} jobs from supabase://{self.BUCKET}/{key}")
        return data

    # ── List ───────────────────────────────────────────────────────────────────

    def list_job_files(self, prefix: str = "jobs/") -> list[str]:
        """
        List all job JSON files under a folder prefix.

        Args:
            prefix: folder name, e.g. "jobs/"

        Returns:
            List of file paths (keys) inside the bucket
        """
        folder = prefix.rstrip("/")
        items = self.client.storage.from_(self.BUCKET).list(folder)
        keys = [f"{folder}/{item['name']}" for item in items if item.get("name")]
        logger.info(f"Found {len(keys)} files under '{prefix}'")
        return keys
