"""
scraper/registry.py
───────────────────
A central registry pattern for managing multiple job scrapers.
To add a new data source in the future, simply build your Scraper class, 
import it here, and add it to the `AVAILABLE_SCRAPERS` list.
"""

from loguru import logger
from typing import Optional

from datetime import datetime, timezone
from typing import Optional
from utils.cloud_storage import CloudStorage

from .remotive_scraper import RemotiveScraper
from .arbeitnow_scraper import ArbeitnowScraper
from .adzuna_scraper import AdzunaScraper
from .wwr_scraper import WWRScraper
from .himalayas_scraper import HimalayasScraper
from .base import JobListing

STALE_HOURS_THRESHOLD = 24

def check_freshness(query: str) -> dict | None:
    """
    Checks if a query was executed recently.
    Returns the cached database result if it's fresh.
    Returns None if it is stale or missing (meaning we must re-scrape).
    """
    storage = CloudStorage()
    logger.info(f"Checking freshness for query: '{query}'...")
    
    response = storage.client.table("rag_outputs").select("*") \
        .eq("query", query).order("created_at", desc=True).limit(1).execute()
        
    data = response.data
    if not data:
        logger.info("No prior execution found. Scraping required.")
        return None
        
    latest_run = data[0]
    created_at_raw = latest_run.get("created_at")
    
    try:
        dt = datetime.fromisoformat(created_at_raw.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        age_hours = (now - dt).total_seconds() / 3600
        
        if age_hours <= STALE_HOURS_THRESHOLD:
            logger.success(f"✨ Fresh data found for '{query}'! ({age_hours:.1f}h old)")
            return latest_run
        else:
            logger.info(f"♻️ Data for '{query}' is stale ({age_hours:.1f}h old). A full re-scrape will trigger.")
            return None
    except Exception as e:
        logger.error(f"Error checking freshness data: {e}. Defaulting to re-scrape.")
        return None

class ScraperRegistry:
    def __init__(self):
        # Register all active scraper classes here
        self.scrapers = [
            RemotiveScraper(),
            ArbeitnowScraper(),
            AdzunaScraper(),
            WWRScraper(),
            HimalayasScraper(),
        ]
        logger.info(f"ScraperRegistry initialized with {len(self.scrapers)} sources.")

    def run_all(
        self,
        query: str,
        location: str = "remote",
        category: Optional[str] = None
    ) -> list[JobListing]:
        """
        Loops through all registered scrapers, aggregates the jobs, and removes duplicates.
        """
        all_jobs = []
        seen_job_ids = set()

        for scraper in self.scrapers:
            try:
                scraper_name = scraper.__class__.__name__
                logger.info(f"▶️ Starting {scraper_name}...")
                
                # Check if the scraper takes a category or location
                # Ensure compatibility across different scraper implementations
                if isinstance(scraper, RemotiveScraper):
                    jobs = scraper.scrape(query=query, location=location, category=category)
                else:
                    jobs = scraper.scrape(query=query)
                
                # Deduplicate by job_id
                for job in jobs:
                    if job.job_id not in seen_job_ids:
                        all_jobs.append(job)
                        seen_job_ids.add(job.job_id)

            except Exception as e:
                logger.error(f"❌ {scraper.__class__.__name__} failed: {e}")

        logger.success(f"✓ ScraperRegistry combined {len(all_jobs)} unique jobs total.")
        return all_jobs
