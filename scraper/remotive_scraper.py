"""
scraper/remotive_scraper.py
────────────────────────────
Fetches remote job listings from the Remotive API.

Remotive provides a free, public JSON API — no authentication needed,
no Cloudflare blocking, and structured data out of the box.

API docs: https://remotive.com/api-documentation
Endpoint: GET https://remotive.com/api/remote-jobs

Params:
  - category:  slug from /api/remote-jobs/categories (e.g. "software-dev", "ai-ml")
  - search:    keyword search across title + description
  - limit:     max results (no hard cap documented, but be reasonable)

Rate limit: max ~4 requests/day recommended by Remotive.
"""

import re
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from loguru import logger


import requests
from loguru import logger

from .base import BaseScraper, JobListing, _html_to_text, _extract_skills


# ─── Category Mapping ─────────────────────────────────────────────────────────

# Maps user-friendly search terms to Remotive category slugs
CATEGORY_MAP = {
    "software":     "software-dev",
    "software-dev": "software-dev",
    "ai":           "ai-ml",
    "ai-ml":        "ai-ml",
    "ml":           "ai-ml",
    "machine learning": "ai-ml",
    "data":         "data",
    "data analysis":"data",
    "devops":       "devops",
    "design":       "design",
    "product":      "product",
    "marketing":    "marketing",
    "qa":           "qa",
    "sales":        "sales-business",
}


# ─── Scraper ───────────────────────────────────────────────────────────────────

class RemotiveScraper(BaseScraper):
    """
    Fetches jobs from the Remotive API.

    Usage:
        scraper = RemotiveScraper()
        listings = scraper.scrape("machine learning engineer")
    """

    API_BASE = "https://remotive.com/api/remote-jobs"

    def __init__(self):
        self.session = requests.Session()
        logger.info("RemotiveScraper ready (API-based, no auth needed)")

    def _fetch(
        self,
        search: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        Fetch jobs from the Remotive API.

        Args:
            search:   Free-text keyword search
            category: Remotive category slug (e.g. "software-dev", "ai-ml")
            limit:    Max results to return

        Returns:
            List of raw job dicts from API
        """
        params = {"limit": limit}
        if search:
            params["search"] = search
        if category:
            params["category"] = category

        logger.info(f"📡 Fetching from Remotive API: {params}")

        try:
            resp = self.session.get(self.API_BASE, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            jobs = data.get("jobs", [])
            total = data.get("job-count", len(jobs))
            logger.info(f"✅ Remotive returned {total} jobs (fetched {len(jobs)})")
            return jobs

        except requests.RequestException as e:
            logger.error(f"Remotive API request failed: {e}")
            return []

    def scrape(
        self,
        query: str,
        location: str = "remote",  # kept for API compatibility, Remotive is all-remote
        max_pages: int = 1,        # kept for API compatibility, not used
        category: Optional[str] = None,
        limit: int = 50,
    ) -> list[JobListing]:
        """
        Fetch jobs matching the query from Remotive.

        Args:
            query:    Search term, e.g. "machine learning engineer"
            location: Ignored (Remotive is remote-only), kept for interface compat
            max_pages: Ignored, kept for interface compatibility
            category: Optional category slug. If None, auto-detects from query.
            limit:    Max results

        Returns:
            List of JobListing dataclass instances
        """
        # Auto-detect category from query if not provided
        if not category:
            query_lower = query.lower()
            for key, slug in CATEGORY_MAP.items():
                if key in query_lower:
                    category = slug
                    logger.info(f"Auto-detected category: {category}")
                    break

        # Fetch from API — do both a search AND category fetch for maximum coverage
        all_jobs = {}  # dedupe by ID

        # Search by keywords
        search_results = self._fetch(search=query, limit=limit)
        for job in search_results:
            all_jobs[job["id"]] = job

        # Also fetch by category if detected
        if category:
            cat_results = self._fetch(category=category, limit=limit)
            for job in cat_results:
                all_jobs[job["id"]] = job

        # If neither search nor category returned results, try broader fetch
        if not all_jobs:
            logger.info("No results from search/category, trying software-dev...")
            fallback = self._fetch(category="software-dev", limit=limit)
            for job in fallback:
                all_jobs[job["id"]] = job

        # Convert to JobListing
        raw_jobs = list(all_jobs.values())
        listings = []
        now = datetime.now(timezone.utc).isoformat()

        for job in raw_jobs:
            description_html = job.get("description", "")
            description_text = _html_to_text(description_html)
            tags = job.get("tags", [])
            skills = _extract_skills(description_text, tags)

            listing = JobListing(
                job_id=str(job.get("id", "")),
                title=job.get("title", "N/A"),
                company=job.get("company_name", "N/A"),
                location=job.get("candidate_required_location", "Remote"),
                description=description_text,
                skills=skills,
                salary=job.get("salary") or None,
                job_url=job.get("url", ""),
                scraped_at=now,
                source="remotive",
            )
            listings.append(listing)
            logger.success(f"  ✓ {listing.title} @ {listing.company}")

        logger.info(f"Scraped {len(listings)} jobs total from Remotive.")
        return listings

    def get_categories(self) -> list[dict]:
        """Fetch available job categories from Remotive."""
        try:
            resp = self.session.get(f"{self.API_BASE}/categories", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("jobs", [])
        except Exception as e:
            logger.error(f"Failed to fetch categories: {e}")
            return []
