"""
scraper/base.py
───────────────
Base utilities and taxonomy for all scrapers.
This prevents the other scrapers from inappropriately importing from 'remotive_scraper.py',
ensuring a clean, uncoupled architecture.
"""

import re
from dataclasses import dataclass
from typing import Optional
from bs4 import BeautifulSoup

@dataclass
class JobListing:
    """Unified job listing model — same fields across all scrapers for pipeline compatibility."""
    job_id: str
    title: str
    company: str
    location: str
    description: str
    skills: list[str]
    salary: Optional[str]
    job_url: str
    scraped_at: str
    source: str

SKILLS_TAXONOMY = {
    # Data & Architecture
    "Python":        ["python", "py"],
    "Java":          ["java", "jvm"],
    "JavaScript":    ["javascript", "js"],
    "TypeScript":    ["typescript", "ts"],
    "SQL":           ["sql", "mysql", "postgresql", "postgres", "t-sql", "plsql"],
    "Spark":         ["apache spark", "pyspark", "spark streaming"],
    "Airflow":       ["apache airflow", "airflow", "dag"],
    "Kafka":         ["apache kafka", "kafka", "kafka streams"],
    "dbt":           ["dbt", "data build tool"],
    "Pandas":        ["pandas"],
    "dask":          ["dask"],
    "Go":            ["golang", "go"],
    "Scala":         ["scala"],
    "C++":           ["c++", "cpp"],
    "Rust":          ["rust"],
    "Swift":         ["swift"],
    "Ruby":          ["ruby", "ruby on rails"],
    "PHP":           ["php", "laravel"],
    
    # Frameworks & Cloud
    "React":         ["react", "react.js", "reactjs"],
    "Next.js":       ["next.js", "nextjs"],
    "Node.js":       ["node.js", "nodejs", "node"],
    "Vue":           ["vue", "vuejs", "vue.js"],
    "Angular":       ["angular"],
    "Kubernetes":    ["kubernetes", "k8s", "kubectl", "helm"],
    "Docker":        ["docker", "dockerfile", "containerisation", "containerization"],
    "Terraform":     ["terraform", "tf", "infrastructure as code", "iac"],
    "AWS":           ["aws", "amazon web services", "s3", "ec2", "glue", "redshift"],
    "GCP":           ["gcp", "google cloud", "bigquery", "dataflow", "pub/sub"],
    "Azure":         ["azure", "microsoft azure", "adls", "synapse"],
    "Snowflake":     ["snowflake"],
    "Databricks":    ["databricks", "delta lake"],
    
    # Analytics & AI
    "Looker":        ["looker", "lookml"],
    "Tableau":       ["tableau"],
    "Power BI":      ["power bi", "powerbi"],
    "Elasticsearch": ["elasticsearch", "elastic search", "elk", "opensearch"],
    "Redis":         ["redis"],
    "MongoDB":       ["mongodb", "mongo"],
    "Machine Learning": ["machine learning", "ml", "ai model", "ai/ml"],
    "Generative AI": ["genai", "llm", "large language model", "chatgpt"],
    "LangChain":     ["langchain"],
    
    # Tooling
    "CI/CD":         ["ci/cd", "github actions", "jenkins"],
    "Figma":         ["figma"],
    "Git":           ["git", "github", "gitlab"],
}

# Aliases that are too short/ambiguous and MUST use word boundaries (\b)
_WORD_BOUNDARY_ALIASES = {"go", "py", "js", "ts", "ml", "tf", "dag", "git", "c++", "llm"}


def _extract_skills(text: str, tags: list[str]) -> list[str]:
    """
    Extract normalized skill names using the unified Skills Taxonomy.
    Groups synonymous words into canonical parent categories.
    """
    text_lower = text.lower()
    found = set()

    for canonical_skill, aliases in SKILLS_TAXONOMY.items():
        for alias in aliases:
            if alias in _WORD_BOUNDARY_ALIASES:
                pattern = r'\b' + re.escape(alias) + r'\b'
                if re.search(pattern, text_lower):
                    found.add(canonical_skill)
                    break
            else:
                if alias in text_lower:
                    found.add(canonical_skill)
                    break
                    
    alias_to_canonical = {}
    for canonical, aliases in SKILLS_TAXONOMY.items():
        for alias in aliases:
            alias_to_canonical[alias] = canonical

    for tag in tags:
        tag_lower = tag.lower().strip()
        if tag_lower in alias_to_canonical:
            found.add(alias_to_canonical[tag_lower])

    return sorted(found)

def _html_to_text(html: str) -> str:
    """Convert HTML description to plain text."""
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n", strip=True)

class BaseScraper:
    """Abstract base class that all scrapers must implement."""
    def scrape(self, query: str, limit: int = 50) -> list[JobListing]:
        raise NotImplementedError("Each scraper must implement the scrape() method.")
