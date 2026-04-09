"""
scraper/arbeitnow_scraper.py
────────────────────────────
Fetches remote job listings from the Arbeitnow API.
Arbeitnow hosts an open job board API which we can use to pad our dataset
alongside Remotive.
"""

import requests
from loguru import logger
from datetime import datetime, timezone
from typing import Optional

# Reuse the exact same data model and skill extraction from Remotive
from .base import BaseScraper, JobListing, _html_to_text, _extract_skills

class ArbeitnowScraper(BaseScraper):
    API_BASE = "https://arbeitnow.com/api/job-board-api"

    def __init__(self):
        self.session = requests.Session()
        # Add basic headers since some simple APIs block python default user agents
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/110.0"
        })
        logger.info("ArbeitnowScraper ready (API-based, no auth needed)")

    def scrape(self, query: str, limit: int = 50) -> list[JobListing]:
        """
        Fetch jobs from Arbeitnow. 
        Since Arbeitnow doesn't have a direct "?search=" parameter on their free API,
        we pull the latest pages and manually regex filter for the query.
        """
        query_lower = query.lower()
        jobs_matched = []
        now = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"📡 Fetching from Arbeitnow API looking for '{query}'...")

        # Fetch up to 3 pages (~300 recent jobs) to find matches
        for page in range(1, 4):
            try:
                resp = self.session.get(f"{self.API_BASE}?page={page}", timeout=15)
                resp.raise_for_status()
                data = resp.json().get("data", [])
                
                for job in data:
                    title = job.get("title", "")
                    description_html = job.get("description", "")
                    description_text = _html_to_text(description_html)
                    
                    # Manually filter down to our search query
                    if query_lower not in title.lower() and query_lower not in description_text.lower():
                        continue
                    
                    tags = job.get("tags", [])
                    skills = _extract_skills(description_text, tags)
                    
                    listing = JobListing(
                        job_id="arbeitnow_" + str(job.get("slug", "")),
                        title=title,
                        company=job.get("company_name", "N/A"),
                        location=job.get("location", "Remote"),
                        description=description_text,
                        skills=skills,
                        salary=None,
                        job_url=job.get("url", ""),
                        scraped_at=now,
                        source="arbeitnow",
                    )
                    jobs_matched.append(listing)
                    logger.success(f"  ✓ (Arbeitnow) {listing.title} @ {listing.company}")
                    
                    if len(jobs_matched) >= limit:
                        break
                        
            except requests.RequestException as e:
                logger.error(f"Arbeitnow API request failed: {e}")
                
            if len(jobs_matched) >= limit:
                break
                
        logger.info(f"Scraped {len(jobs_matched)} jobs total from Arbeitnow.")
        return jobs_matched
