#!/usr/bin/env python3
"""
Utility script to analyze downloaded NVIDIA forum questions.
Provides statistics and search functionality for the downloaded JSON files.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Dict


class ForumAnalyzer:
    """Analyzer for downloaded forum questions."""
    
    def __init__(self, questions_dir: str = "all_questions"):
        """Initialize analyzer with questions directory."""
        self.questions_dir = Path(questions_dir)
        self.questions = []
        self.load_questions()
    
    def load_questions(self):
        """Load all JSON files from the questions directory."""
        if not self.questions_dir.exists():
            print(f"Error: Directory '{self.questions_dir}' not found.")
            return
        
        json_files = list(self.questions_dir.glob("*.json"))
        print(f"Loading {len(json_files)} questions...")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.questions.append(data)
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
        
        print(f"Successfully loaded {len(self.questions)} questions.\n")
    
    def show_statistics(self):
        """Display statistics about the downloaded questions."""
        if not self.questions:
            print("No questions loaded.")
            return
        
        print("=" * 70)
        print("STATISTICS")
        print("=" * 70)
        
        # Basic counts
        total = len(self.questions)
        total_posts = sum(q.get('posts_count', 0) for q in self.questions)
        total_views = sum(q.get('views', 0) for q in self.questions)
        total_likes = sum(q.get('like_count', 0) for q in self.questions)
        
        print(f"\nTotal Questions: {total}")
        print(f"Total Posts: {total_posts}")
        print(f"Total Views: {total_views:,}")
        print(f"Total Likes: {total_likes}")
        print(f"Average Posts per Question: {total_posts / total:.1f}")
        print(f"Average Views per Question: {total_views / total:.1f}")
        
        # Date range
        dates = [q.get('created_at') for q in self.questions if q.get('created_at')]
        if dates:
            dates_sorted = sorted(dates)
            first_date = datetime.fromisoformat(dates_sorted[0].replace('Z', '+00:00'))
            last_date = datetime.fromisoformat(dates_sorted[-1].replace('Z', '+00:00'))
            print(f"\nDate Range: {first_date.date()} to {last_date.date()}")
        
        # Most viewed questions
        print("\n" + "-" * 70)
        print("TOP 5 MOST VIEWED QUESTIONS")
        print("-" * 70)
        most_viewed = sorted(self.questions, key=lambda x: x.get('views', 0), reverse=True)[:5]
        for i, q in enumerate(most_viewed, 1):
            print(f"{i}. [{q.get('views', 0):,} views] {q.get('title', 'N/A')}")
        
        # Most discussed questions
        print("\n" + "-" * 70)
        print("TOP 5 MOST DISCUSSED QUESTIONS")
        print("-" * 70)
        most_posts = sorted(self.questions, key=lambda x: x.get('posts_count', 0), reverse=True)[:5]
        for i, q in enumerate(most_posts, 1):
            print(f"{i}. [{q.get('posts_count', 0)} posts] {q.get('title', 'N/A')}")
        
        # Tags analysis
        print("\n" + "-" * 70)
        print("TOP 10 TAGS")
        print("-" * 70)
        all_tags = []
        for q in self.questions:
            tags = q.get('tags', [])
            if isinstance(tags, list):
                all_tags.extend(tags)
        
        if all_tags:
            tag_counts = Counter(all_tags)
            for i, (tag, count) in enumerate(tag_counts.most_common(10), 1):
                print(f"{i}. {tag}: {count} questions")
        else:
            print("No tags found.")
        
        print("\n" + "=" * 70 + "\n")
    
    def search(self, keyword: str):
        """Search for questions containing the keyword."""
        keyword_lower = keyword.lower()
        results = []
        
        for q in self.questions:
            # Search in title
            if keyword_lower in q.get('title', '').lower():
                results.append(('title', q))
                continue
            
            # Search in posts
            posts = q.get('post_stream', {}).get('posts', [])
            for post in posts:
                post_text = post.get('cooked', '')
                if keyword_lower in post_text.lower():
                    results.append(('post', q))
                    break
        
        print(f"\nFound {len(results)} questions matching '{keyword}':")
        print("-" * 70)
        
        for i, (match_type, q) in enumerate(results, 1):
            title = q.get('title', 'N/A')
            topic_id = q.get('id', 'N/A')
            slug = q.get('slug', 'N/A')
            url = f"https://forums.developer.nvidia.com/t/{slug}/{topic_id}"
            print(f"\n{i}. {title}")
            print(f"   URL: {url}")
            print(f"   Views: {q.get('views', 0):,} | Posts: {q.get('posts_count', 0)} | Match: {match_type}")
        
        print("\n" + "=" * 70 + "\n")
    
    def list_recent(self, limit: int = 10):
        """List most recent questions."""
        sorted_questions = sorted(
            self.questions,
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )[:limit]
        
        print(f"\n{limit} MOST RECENT QUESTIONS:")
        print("-" * 70)
        
        for i, q in enumerate(sorted_questions, 1):
            title = q.get('title', 'N/A')
            created = q.get('created_at', 'N/A')
            if created != 'N/A':
                date = datetime.fromisoformat(created.replace('Z', '+00:00'))
                created = date.strftime('%Y-%m-%d')
            print(f"{i}. [{created}] {title}")
        
        print("\n" + "=" * 70 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze downloaded NVIDIA forum questions"
    )
    parser.add_argument(
        '-d', '--dir',
        default='all_questions',
        help='Directory containing JSON files (default: all_questions)'
    )
    parser.add_argument(
        '-s', '--stats',
        action='store_true',
        help='Show statistics'
    )
    parser.add_argument(
        '-q', '--search',
        type=str,
        help='Search for keyword in questions'
    )
    parser.add_argument(
        '-r', '--recent',
        type=int,
        default=10,
        help='List N most recent questions (default: 10)'
    )
    parser.add_argument(
        '-l', '--list-recent',
        action='store_true',
        help='List most recent questions'
    )
    
    args = parser.parse_args()
    
    analyzer = ForumAnalyzer(questions_dir=args.dir)
    
    if not analyzer.questions:
        print("No questions found. Have you run the download script?")
        return
    
    # If no specific action, show stats by default
    if not (args.stats or args.search or args.list_recent):
        args.stats = True
    
    if args.stats:
        analyzer.show_statistics()
    
    if args.search:
        analyzer.search(args.search)
    
    if args.list_recent:
        analyzer.list_recent(args.recent)


if __name__ == "__main__":
    main()
