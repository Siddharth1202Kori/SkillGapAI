"""
scraper/wwr_scraper.py
───────────────────────
Fetches remote job listings from WeWorkRemotely RSS feed.
RSS feeds are extremely reliable and never block IP addresses.
"""

import feedparser
from loguru import logger
from datetime import datetime, timezone
from bs4 import BeautifulSoup

from .base import BaseScraper, JobListing, _extract_skills

class WWRScraper(BaseScraper):
    # WWR's universal global feed (recent jobs across all categories)
    FEED_URL = "https://weworkremotely.com/remote-jobs.rss"

    def __init__(self):
        logger.info("WWRScraper ready (RSS-based, no auth needed)")

    def scrape(self, query: str, limit: int = 50) -> list[JobListing]:
        query_lower = query.lower()
        jobs_matched = []
        now = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"📡 Fetching from WeWorkRemotely RSS looking for '{query}'...")
        feed = feedparser.parse(self.FEED_URL)
        
        for entry in feed.entries:
            title = entry.get('title', '')
            link = entry.get('link', '')
            description_html = entry.get('description', '')
            
            # WWR embeds full HTML in the description node
            soup = BeautifulSoup(description_html, "lxml")
            description_text = soup.get_text(separator="\n", strip=True)
            
            # Client-side filtering
            if query_lower not in title.lower() and query_lower not in description_text.lower():
                continue
                
            skills = _extract_skills(description_text, [])
            author = entry.get('author', '')
            # WWR RSS often prefixes author as "Company Name", we clean it up if possible
            company = author.split(" : ")[-1] if author else "WWR Company"
            
            listing = JobListing(
                job_id="wwr_" + str(entry.get("id", link)),
                title=title.replace(f"{company}: ", ""),  # Strip company name from title
                company=company,
                location="Remote",
                description=description_text,
                skills=skills,
                salary=None,
                job_url=link,
                scraped_at=now,
                source="weworkremotely"
            )
            jobs_matched.append(listing)
            logger.success(f"  ✓ (WWR) {listing.title} @ {listing.company}")
            
            if len(jobs_matched) >= limit:
                break
                
        logger.info(f"Scraped {len(jobs_matched)} jobs from WeWorkRemotely.")
        return jobs_matched
