import cloudscraper
from bs4 import BeautifulSoup
import time
import random
import csv
import re
import os

# --- Configuration ---
BASE_URL = "https://mtgdecks.net/Commander/date-6"
OUTPUT_FILE = "mtg_commander_data.csv"
DECKS_PER_COMMANDER = 5
# Limit how many commanders to scrape 
COMMANDER_LIMIT = 300 

def get_soup(scraper, url):
    """Fetches HTML content using a persistent cloudscraper session."""
    try:
        response = scraper.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def get_commander_links(scraper, base_url):
    """Extracts links to individual Commander pages from the main tier list."""
    print(f"Fetching top Commanders from {base_url}...")
    soup = get_soup(scraper, base_url)
    if not soup:
        return []

    commander_links = []
    # Find all links on the main page
    elements = soup.find_all('a', href=True)
    
    for element in elements:
        href = element['href']
        # Filter for Commander pages
        if href.startswith('/Commander/') and '/deck-' not in href and href != '/Commander/tournaments':
            full_url = f"https://mtgdecks.net{href}"
            commander_links.append(full_url)
            
    # Use a set to remove duplicates
    unique_commanders = list(set(commander_links))
    return unique_commanders

def get_deck_links(scraper, commander_url):
    """Extracts up to a specified number of deck links for a specific Commander."""
    soup = get_soup(scraper, commander_url)
    if not soup:
        return []

    deck_links = []
    elements = soup.find_all('a', href=True) 
    
    for element in elements:
        href = element['href']
        if '/Commander/deck-decklist-by' in href: 
            full_url = f"https://mtgdecks.net{href}"
            deck_links.append(full_url)
            
    unique_decks = list(set(deck_links))
    return unique_decks[:DECKS_PER_COMMANDER]

def scrape_deck_data(scraper, deck_url, commander_name):
    """Scrapes card data from an individual deck page."""
    print(f"  -> Scraping deck: {deck_url}")
    soup = get_soup(scraper, deck_url)
    if not soup:
        return []

    deck_data = []
    card_rows = soup.find_all('tr', class_='cardItem')

    for row in card_rows:
        try:
            card_name = row.get('data-card-id')
            rarity = row.get('data-rarity', 'Unknown')
            
            price = row.get('tcgplayer') or row.get('cardkingdom') or "0.00"
            
            a_tag = row.find('a')
            card_type = a_tag.get('type') if a_tag else "Unknown"

            if card_name:
                deck_data.append({
                    "Commander": commander_name,
                    "Deck_URL": deck_url,
                    "Card_Name": card_name,
                    "Card_Type": card_type,
                    "Rarity": rarity,
                    "Price_USD": float(price)
                })
            
        except Exception as e:
            print(f"Error parsing a card row in {deck_url}: {e}")
            continue

    return deck_data

def main():
    all_cards_data = []
    
    # Initialize a single scraper session for the entire run
    scraper = cloudscraper.create_scraper(browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    })
    
    commander_links = get_commander_links(scraper, BASE_URL)

    print(f"Found {len(commander_links)} Commanders. Limiting to {COMMANDER_LIMIT} for this run.")
    
    # Iterate through Commanders
    for cmdr_link in commander_links[:COMMANDER_LIMIT]:
        # Extract commander name from URL
        cmdr_name = cmdr_link.split('/')[-1].replace('-', ' ').title()
        print(f"\nProcessing Commander: {cmdr_name}")
        
        # Get deck variations
        deck_links = get_deck_links(scraper, cmdr_link)
        
        # Scrape each deck
        for deck_link in deck_links:
            deck_cards = scrape_deck_data(scraper, deck_link, cmdr_name)
            all_cards_data.extend(deck_cards)
            
            # Rate limit between individual deck requests
            time.sleep(random.uniform(2.0, 4.0))
            
        # Rate limit between Commander pages
        time.sleep(random.uniform(3.0, 6.0))

    # Export to CSV
    if all_cards_data:
        keys = all_cards_data[0].keys()
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_cards_data)
        print(f"\nSuccess! Scraped {len(all_cards_data)} total cards across multiple decks. Saved to {OUTPUT_FILE}")
    else:
        print("\nNo data was scraped. Double check the URLs or HTML structure.")

if __name__ == "__main__":
    main()