"""
scraper/adzuna_scraper.py
──────────────────────────
Fetches job listings from the Adzuna API.
Unlike Remotive/Arbeitnow, Adzuna requires authentication.
You must set ADZUNA_APP_ID and ADZUNA_APP_KEY in your .env file!
"""

import os
import requests
from loguru import logger
from datetime import datetime, timezone

from .base import BaseScraper, JobListing, _html_to_text, _extract_skills

class AdzunaScraper(BaseScraper):
    # We default to the US endpoint for broad remote jobs, but this can be parametrized
    API_BASE = "https://api.adzuna.com/v1/api/jobs/us/search"

    def __init__(self):
        self.app_id = os.getenv("ADZUNA_APP_ID")
        self.app_key = os.getenv("ADZUNA_APP_KEY")
        self.session = requests.Session()
        
        if not self.app_id or not self.app_key:
            logger.warning("AdzunaScraper disabled: ADZUNA_APP_ID or ADZUNA_APP_KEY not set in .env")
        else:
            logger.info("AdzunaScraper ready (Auth keys loaded)")

    def scrape(self, query: str, limit: int = 50) -> list[JobListing]:
        """Fetch jobs from Adzuna API."""
        if not self.app_id or not self.app_key:
            return []

        jobs_matched = []
        now = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"📡 Fetching from Adzuna API for '{query}'...")

        params = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "results_per_page": min(limit, 50),
            "what": query,
            "where": "remote", # Try to enforce remote jobs
            "content-type": "application/json"
        }

        try:
            # Fetch page 1
            resp = self.session.get(f"{self.API_BASE}/1", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("results", [])
            
            for job in data:
                title = job.get("title", "")
                company_obj = job.get("company", {})
                company_name = company_obj.get("display_name", "N/A")
                
                description_html = job.get("description", "")
                description_text = _html_to_text(description_html)
                
                # Adzuna doesn't provide tags, so we rely heavily on our keyword extractor
                skills = _extract_skills(description_text, [])
                
                location_obj = job.get("location", {})
                location = location_obj.get("display_name", "Remote")
                
                salary_min = job.get("salary_min")
                salary_max = job.get("salary_max")
                salary_str = f"${int(salary_min)} - ${int(salary_max)}" if salary_min and salary_max else None

                listing = JobListing(
                    job_id="adzuna_" + str(job.get("id", "")),
                    title=title,
                    company=company_name,
                    location=location,
                    description=description_text,
                    skills=skills,
                    salary=salary_str,
                    job_url=job.get("redirect_url", ""),
                    scraped_at=now,
                    source="adzuna",
                )
                jobs_matched.append(listing)
                logger.success(f"  ✓ (Adzuna) {listing.title} @ {listing.company}")
                    
        except requests.RequestException as e:
            logger.error(f"Adzuna API request failed: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Adzuna response: {e.response.text}")
                
        logger.info(f"Scraped {len(jobs_matched)} jobs from Adzuna.")
        return jobs_matched
