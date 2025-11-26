#!/usr/bin/env python3
"""
Script to download all questions from the NVIDIA DGX Spark GB10 forum.
Each question is saved as a separate JSON file.
"""

import json
import os
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin


class NvidiaForumScraper:
    """Scraper for NVIDIA Developer Forum questions."""
    
    BASE_URL = "https://forums.developer.nvidia.com"
    CATEGORY_ID = "719"
    LIST_ENDPOINT = f"/c/accelerated-computing/dgx-spark-gb10/{CATEGORY_ID}/l/latest.json"
    
    def __init__(self, output_dir: str = "all_questions"):
        """
        Initialize the scraper.
        
        Args:
            output_dir: Directory to save downloaded JSON files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
    def get_topics_page(self, page: int = 1) -> Optional[Dict]:
        """
        Fetch a page of topics.
        
        Args:
            page: Page number to fetch
            
        Returns:
            JSON response or None if error
        """
        url = urljoin(self.BASE_URL, self.LIST_ENDPOINT)
        params = {
            'filter': 'default',
            'solved':'yes',
            'page': page
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            return None
    
    def get_topic_detail(self, topic_id: int, slug: str) -> Optional[Dict]:
        """
        Fetch detailed information for a specific topic.
        
        Args:
            topic_id: Topic ID
            slug: Topic slug (URL-friendly title)
            
        Returns:
            JSON response or None if error
        """
        url = f"{self.BASE_URL}/t/{slug}/{topic_id}.json"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching topic {topic_id}: {e}")
            return None
    
    def save_topic(self, topic_data: Dict, topic_id: int, slug: str) -> None:
        """
        Save topic data to a JSON file.
        
        Args:
            topic_data: Topic data to save
            topic_id: Topic ID
            slug: Topic slug
        """
        # Create a safe filename using topic_id and slug
        filename = f"{topic_id}_{slug}.json"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(topic_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved: {filename}")
        except IOError as e:
            print(f"Error saving {filename}: {e}")
    
    def extract_topics_from_page(self, page_data: Dict) -> List[Dict]:
        """
        Extract topic information from a page response.
        
        Args:
            page_data: JSON response from topics list
            
        Returns:
            List of topic dictionaries with id, slug, and title
        """
        topics = []
        if 'topic_list' in page_data and 'topics' in page_data['topic_list']:
            for topic in page_data['topic_list']['topics']:
                topics.append({
                    'id': topic['id'],
                    'slug': topic['slug'],
                    'title': topic.get('title', ''),
                    'views': topic.get('views', 0),
                    'posts_count': topic.get('posts_count', 0)
                })
        return topics
    
    def scrape_all(self, delay: float = 1.0, max_pages: Optional[int] = None) -> int:
        """
        Scrape all questions from the forum.
        
        Args:
            delay: Delay between requests in seconds (be respectful!)
            max_pages: Maximum number of pages to scrape (None for all)
            
        Returns:
            Total number of topics downloaded
        """
        print(f"Starting download to: {self.output_dir.absolute()}")
        print("-" * 70)
        
        page = 1
        total_downloaded = 0
        
        while True:
            if max_pages and page > max_pages:
                print(f"\nReached max pages limit ({max_pages})")
                break
            
            print(f"\nFetching page {page}...")
            page_data = self.get_topics_page(page)
            
            if not page_data:
                print("Failed to fetch page data. Stopping.")
                break
            
            topics = self.extract_topics_from_page(page_data)
            
            if not topics:
                print("No more topics found. Done!")
                break
            
            print(f"Found {len(topics)} topics on page {page}")
            
            # Download each topic
            for i, topic in enumerate(topics, 1):
                # Check if file already exists to avoid re-downloading
                filename = f"{topic['id']}_{topic['slug']}.json"
                filepath = self.output_dir / filename
                
                if filepath.exists():
                    print(f"  [{i}/{len(topics)}] ⏭️  Skipping (already exists): {topic['title'][:60]}...")
                    continue

                print(f"  [{i}/{len(topics)}] Downloading: {topic['title'][:60]}...")
                
                topic_detail = self.get_topic_detail(topic['id'], topic['slug'])
                
                if topic_detail:
                    self.save_topic(topic_detail, topic['id'], topic['slug'])
                    total_downloaded += 1
                else:
                    print(f"  ✗ Failed to download topic {topic['id']}")
                
                # Be respectful with rate limiting
                time.sleep(delay)
            
            # Check if there's a next page
            topic_list = page_data.get('topic_list', {})
            more_topics_url = topic_list.get('more_topics_url')
            
            if not more_topics_url:
                print("\nNo more pages available. Done!")
                break
            
            page += 1
            time.sleep(delay)  # Additional delay between pages
        
        print("-" * 70)
        print(f"\n✓ Download complete! Total topics saved: {total_downloaded}")
        print(f"Files saved to: {self.output_dir.absolute()}")
        
        return total_downloaded


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download all questions from NVIDIA DGX Spark GB10 forum"
    )
    parser.add_argument(
        '-o', '--output',
        default='all_questions',
        help='Output directory for JSON files (default: all_questions)'
    )
    parser.add_argument(
        '-d', '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    parser.add_argument(
        '-m', '--max-pages',
        type=int,
        default=None,
        help='Maximum number of pages to scrape (default: all pages)'
    )
    
    args = parser.parse_args()
    
    scraper = NvidiaForumScraper(output_dir=args.output)
    
    try:
        total = scraper.scrape_all(delay=args.delay, max_pages=args.max_pages)
        print(f"\n{'=' * 70}")
        print(f"Summary: Successfully downloaded {total} questions")
        print(f"{'=' * 70}")
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
