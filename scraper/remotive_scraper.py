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


# ─── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class JobListing:
    """Unified job listing model — same fields as IndeedScraper for compatibility."""
    job_id: str
    title: str
    company: str
    location: str
    description: str
    skills: list[str]
    salary: Optional[str]
    job_url: str
    scraped_at: str
    source: str = "remotive"


# ─── Skill Extraction ─────────────────────────────────────────────────────────

SKILL_KEYWORDS = [
    "python", "java", "javascript", "typescript", "react", "node.js", "nodejs",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform",
    "machine learning", "deep learning", "pytorch", "tensorflow", "scikit-learn",
    "langchain", "openai", "hugging face", "huggingface",
    "fastapi", "django", "flask", "spring", "express.js",
    "ci/cd", "jenkins", "github actions",
    "rest api", "graphql", "microservices", "kafka",
    "pandas", "numpy", "spark", "hadoop", "airflow",
    "linux", "bash", "shell scripting",
    "golang", "rust", "c++", "ruby", "php", "swift", "kotlin",
    "next.js", "vue", "angular", "svelte",
    "figma", "tailwind",
]

# Short keywords that need word-boundary matching to avoid false positives
# e.g., "go" matching "go to the website", "r" matching "your", "git" matching "legitimate"
_WORD_BOUNDARY_KEYWORDS = ["go", "r", "css", "rag", "llm", "git"]


def _extract_skills(text: str, tags: list[str]) -> list[str]:
    """
    Extract skills from description text + API tags.
    Tags from Remotive already contain skill-like data, so we merge both.
    """
    text_lower = text.lower()
    found = set()

    # Standard substring matching for multi-word / long keywords
    for skill in SKILL_KEYWORDS:
        if skill in text_lower:
            found.add(skill)

    # Word-boundary matching for short/ambiguous keywords
    for skill in _WORD_BOUNDARY_KEYWORDS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill)

    # Also include relevant tags (these are curated by Remotive, so trust them more)
    all_known = set(s.lower() for s in SKILL_KEYWORDS + _WORD_BOUNDARY_KEYWORDS)
    for tag in tags:
        tag_lower = tag.lower().strip()
        if tag_lower in all_known:
            found.add(tag_lower)
        # Also match compound tags like "AI/ML"
        if tag_lower in ("ai/ml", "ai", "ml"):
            found.add("machine learning")

    return sorted(found)


def _html_to_text(html: str) -> str:
    """Convert HTML description to plain text."""
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n", strip=True)


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

class RemotiveScraper:
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
