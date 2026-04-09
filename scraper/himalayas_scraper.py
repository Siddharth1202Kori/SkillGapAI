"""
scraper/himalayas_scraper.py
────────────────────────────
Fetches remote jobs from Himalayas.app API.
"""

import requests
from loguru import logger
from datetime import datetime, timezone

from .base import BaseScraper, JobListing, _html_to_text, _extract_skills

class HimalayasScraper(BaseScraper):
    API_BASE = "https://himalayas.app/jobs/api"

    def __init__(self):
        self.session = requests.Session()
        logger.info("HimalayasScraper ready (API-based, no auth needed)")

    def scrape(self, query: str, limit: int = 50) -> list[JobListing]:
        query_lower = query.lower()
        jobs_matched = []
        now = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"📡 Fetching from Himalayas API for '{query}'...")
        
        # Paginate through the first 400 recent jobs. It's fast to run 4 requests.
        for limit_param, offset in [(100, 0), (100, 100), (100, 200), (100, 300)]:
            try:
                resp = self.session.get(f"{self.API_BASE}?limit={limit_param}&offset={offset}", timeout=15)
                resp.raise_for_status()
                data = resp.json().get("jobs", [])
                
                if not data:
                    break  # Hit the end early
                    
                for job in data:
                    title = job.get("title", "")
                    description_html = job.get("description", "")
                    description_text = _html_to_text(description_html)
                    
                    if query_lower not in title.lower() and query_lower not in description_text.lower():
                        continue
                        
                    tags = job.get("categories", [])
                    skills = _extract_skills(description_text, [str(t) for t in tags])
                    
                    # Some roles have location restrictions (e.g. "US Only"). Default to Remote.
                    restrictions = job.get("locationRestrictions", [])
                    location = ", ".join(restrictions) if restrictions else "Remote"
                    
                    # Extract salary if provided
                    salary = None
                    if job.get("minSalary") and job.get("maxSalary"):
                        salary = f"${job.get('minSalary')} - ${job.get('maxSalary')}"
                    
                    listing = JobListing(
                        job_id="himalayas_" + str(job.get("id", job.get("url", ""))),
                        title=title,
                        company=job.get("companyName", "N/A"),
                        location=location,
                        description=description_text,
                        skills=skills,
                        salary=salary,
                        job_url=job.get("applicationLink", job.get("url", "")),
                        scraped_at=now,
                        source="himalayas"
                    )
                    jobs_matched.append(listing)
                    logger.success(f"  ✓ (Himalayas) {listing.title} @ {listing.company}")
                    
                    if len(jobs_matched) >= limit:
                        break
                        
            except requests.RequestException as e:
                logger.error(f"Himalayas API request failed: {e}")
                
            if len(jobs_matched) >= limit:
                break
                
        logger.info(f"Scraped {len(jobs_matched)} jobs from Himalayas.")
        return jobs_matched
