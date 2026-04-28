#!/usr/bin/env python3
"""
MTG Commander Deck Scraper
Scrapes deck data from mtgdecks.net/Commander.

Authors: Aaron Wohleb, Wil Lehan, Jac Dreifurst
"""

import cloudscraper
from bs4 import BeautifulSoup
import time
import random
import csv
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

#  Configuration
BASE_URL = "https://mtgdecks.net/Commander/date-6"
OUTPUT_FILE = "mtg_commander_data.csv"
CHECKPOINT_FILE = "scrape_checkpoint.json"
DECKS_PER_COMMANDER = 7
COMMANDER_LIMIT = 2000

# Concurrency 
WORKERS = 3                # concurrent commander grabbers
DECK_WORKERS = 5           # concurrent deck fetches per commander

# Rate limiting 
MIN_DELAY = 0.4
MAX_DELAY = 0.8

# Retry
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0        # exponential base (2s, 4s, 8s)

#  Thread-safe rate limiter
class RateLimiter:
    """
    Ensures that across all threads, requests are spaced at least `min_gap` seconds apart.
    """
    def __init__(self, min_gap: float = 0.3):
        self._lock = threading.Lock()
        self._last = 0.0
        self._min_gap = min_gap

    def wait(self):
        with self._lock:
            now = time.monotonic()
            deadline = self._last + self._min_gap
            if now < deadline:
                time.sleep(deadline - now)
            self._last = time.monotonic()


_rate = RateLimiter(min_gap=0.35)

#  HTTP helpers
def make_scraper():
    """Create a cloudscraper session (one per thread)."""
    return cloudscraper.create_scraper(browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True,
    })


# Thread-local storage so each thread reuses its own session.
_local = threading.local()

def get_scraper():
    if not hasattr(_local, 'scraper'):
        _local.scraper = make_scraper()
    return _local.scraper


def fetch_soup(url: str) -> BeautifulSoup | None:
    """
    GET a URL with retry and exponential backoff.
    Returns a BeautifulSoup object or None.
    """
    scraper = get_scraper()

    for attempt in range(1, MAX_RETRIES + 1):
        _rate.wait()
        try:
            resp = scraper.get(url, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, 'lxml')
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF ** attempt + random.uniform(0, 1)
                print(f"  Retry {attempt}/{MAX_RETRIES} for {url} "
                      f"({e}) , waiting {wait:.1f}s")
                time.sleep(wait)
            else:
                print(f" Failed after {MAX_RETRIES} attempts: {url} ({e})")
                return None


#  Parsing
def get_commander_links(base_url: str) -> list[str]:
    """Extract Commander page URLs from the main list."""
    print(f"Fetching commander list from {base_url} …")
    soup = fetch_soup(base_url)
    if not soup:
        return []

    seen = set()
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if (href.startswith('/Commander/')
                and '/deck-' not in href
                and href != '/Commander/tournaments'):
            url = f"https://mtgdecks.net{href}"
            if url not in seen:
                seen.add(url)
                links.append(url)
    return links


def get_deck_links(commander_url: str) -> list[str]:
    """Extract deck-list URLs from a Commander's page."""
    soup = fetch_soup(commander_url)
    if not soup:
        return []

    seen = set()
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '/Commander/deck-decklist-by' in href:
            url = f"https://mtgdecks.net{href}"
            if url not in seen:
                seen.add(url)
                links.append(url)
    return links[:DECKS_PER_COMMANDER]


def scrape_deck(deck_url: str, commander_name: str) -> list[dict]:
    """Parse card data from a single deck page."""
    soup = fetch_soup(deck_url)
    if not soup:
        return []

    cards = []
    for row in soup.find_all('tr', class_='cardItem'):
        try:
            name = row.get('data-card-id')
            if not name:
                continue
            rarity = row.get('data-rarity', 'Unknown')
            price = row.get('tcgplayer') or row.get('cardkingdom') or "0.00"
            a_tag = row.find('a')
            card_type = a_tag.get('type') if a_tag else "Unknown"
            cards.append({
                "Commander": commander_name,
                "Deck_URL": deck_url,
                "Card_Name": name,
                "Card_Type": card_type,
                "Rarity": rarity,
                "Price_USD": float(price),
            })
        except Exception as e:
            print(f" Parse error in {deck_url}: {e}")
    return cards


#  Per-commander logic
def process_commander(cmdr_url: str) -> list[dict]:
    """
      Fetchs the commander page to get deck links.
      Fetch all decks concurrently.
      Return the combined card list.
    """
    cmdr_name = cmdr_url.split('/')[-1].replace('-', ' ').title()

    deck_urls = get_deck_links(cmdr_url)
    if not deck_urls:
        return []

    all_cards = []

    # Fetch decks for this commander in parallel
    with ThreadPoolExecutor(max_workers=DECK_WORKERS) as pool:
        futures = {
            pool.submit(scrape_deck, url, cmdr_name): url
            for url in deck_urls
        }
        for fut in as_completed(futures):
            try:
                all_cards.extend(fut.result())
            except Exception as e:
                print(f" Deck thread error: {e}")

    return all_cards


#  Checkpoint helpers
def load_checkpoint() -> set[str]:
    """Return the set of commander URLs already scraped."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            done = set(data.get("completed", []))
            print(f"Resuming, {len(done)} commanders already done.")
            return done
    return set()


def save_checkpoint(completed: set[str]):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"completed": list(completed)}, f)


#  CSV incremental writer 
FIELDNAMES = ["Commander", "Deck_URL", "Card_Name",
              "Card_Type", "Rarity", "Price_USD"]

_csv_lock = threading.Lock()

def init_csv(path: str, resume: bool):
    """Create the CSV with a header (or leave it if resuming)."""
    if resume and os.path.exists(path):
        return   # keep existing data
    with open(path, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def append_csv(path: str, rows: list[dict]):
    """Append rows to the CSV (thread-safe)."""
    if not rows:
        return
    with _csv_lock:
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerows(rows)


#  Main
def main():
    print("=" * 60)
    print("  MTG COMMANDER DECK SCRAPER")
    print("=" * 60)
    print(f"  Workers:           {WORKERS} commanders in parallel")
    print(f"  Deck workers:      {DECK_WORKERS} decks per commander")
    print(f"  Decks/commander:   {DECKS_PER_COMMANDER}")
    print(f"  Commander limit:   {COMMANDER_LIMIT}")
    print(f"  Output:            {OUTPUT_FILE}")
    print()

    # Get all commander URLs
    all_cmdr_links = get_commander_links(BASE_URL)
    if not all_cmdr_links:
        print("No commander links found.")
        sys.exit(1)

    cmdr_links = all_cmdr_links[:COMMANDER_LIMIT]
    print(f"Found {len(all_cmdr_links)} commanders, "
          f"processing {len(cmdr_links)}.\n")

    # Load checkpoint
    completed = load_checkpoint()
    remaining = [url for url in cmdr_links if url not in completed]
    resuming = len(completed) > 0

    init_csv(OUTPUT_FILE, resume=resuming)

    print(f"Remaining: {len(remaining)} commanders\n")
    if not remaining:
        print("all commanders already scraped. Delete "
              f"{CHECKPOINT_FILE} to restart.")
        return

    # Process commanders
    t0 = time.time()
    total_cards = 0
    done_count = len(completed)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(process_commander, url): url
            for url in remaining
        }

        for fut in as_completed(futures):
            url = futures[fut]
            cmdr_name = url.split('/')[-1].replace('-', ' ').title()
            try:
                cards = fut.result()
                append_csv(OUTPUT_FILE, cards)
                total_cards += len(cards)
                completed.add(url)
                save_checkpoint(completed)
                done_count += 1

                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (len(cmdr_links) - done_count) / rate if rate > 0 else 0
                print(f"[{done_count}/{len(cmdr_links)}] "
                      f"{cmdr_name}: {len(cards)} cards  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

            except Exception as e:
                print(f"error: {cmdr_name}: {e}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Done, {total_cards:,} cards scraped in {elapsed:.0f}s "
          f"({elapsed/60:.1f} min)")
    print(f"  Saved to {OUTPUT_FILE}")
    print(f"  Checkpoint: {CHECKPOINT_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()