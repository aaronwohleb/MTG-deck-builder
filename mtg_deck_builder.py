"""
MTG Commander Deck Builder
Uses Item-to-Item Collaborative Filtering (cosine similarity on card
co-occurrence) to recommend cards that complete a Commander deck.

FEATURES:
- Uses color identity enforcement via Scryfall API 
- Commander-weighted scoring
- Uses Card-type quotas 
- Scryfall lookup for cards not in the local database
- Optional budget cap

Authors: Aaron Wohleb, Wil Lehan, Jac Dreifurst
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from collections import Counter
from difflib import get_close_matches

#  Constants
DECK_SIZE = 100
SCRYFALL_CACHE_FILE = "scryfall_cache.json"
SCRYFALL_BASE = "https://api.scryfall.com"

# Commander pool gets 80% of the scoring weight
COMMANDER_WEIGHT = 0.80
SEED_WEIGHT = 0.20

# Min / max slots per broad card category
TYPE_TARGETS = {
    "land":         (33, 40),
    "creature":     (20, 35),
    "instant":      (8, 18),
    "sorcery":      (4, 12),
    "artifact":     (6, 16),
    "enchantment":  (3, 10),
    "planeswalker": (0, 3),
    "other":        (0, 5),
}

# Maps well-known lands to colors.  Used as offline fallback.
_LAND_COLOR_MAP = {
    "Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G",
    "Snow-Covered Plains": "W", "Snow-Covered Island": "U",
    "Snow-Covered Swamp": "B", "Snow-Covered Mountain": "R",
    "Snow-Covered Forest": "G",
    "Hallowed Fountain": "WU", "Watery Grave": "UB", "Blood Crypt": "BR",
    "Stomping Ground": "RG", "Temple Garden": "GW", "Godless Shrine": "WB",
    "Steam Vents": "UR", "Overgrown Tomb": "BG", "Sacred Foundry": "RW",
    "Breeding Pool": "GU",
    "Tundra": "WU", "Underground Sea": "UB", "Badlands": "BR",
    "Taiga": "RG", "Savannah": "GW", "Scrubland": "WB",
    "Volcanic Island": "UR", "Bayou": "BG", "Plateau": "RW",
    "Tropical Island": "GU",
    "Glacial Fortress": "WU", "Drowned Catacomb": "UB",
    "Dragonskull Summit": "BR", "Rootbound Crag": "RG",
    "Sunpetal Grove": "GW", "Isolated Chapel": "WB",
    "Sulfur Falls": "UR", "Woodland Cemetery": "BG",
    "Clifftop Retreat": "RW", "Hinterland Harbor": "GU",
    "Seachrome Coast": "WU", "Darkslick Shores": "UB",
    "Blackcleave Cliffs": "BR", "Copperline Gorge": "RG",
    "Razorverge Thicket": "GW", "Concealed Courtyard": "WB",
    "Spirebluff Canal": "UR", "Blooming Marsh": "BG",
    "Inspiring Vantage": "RW", "Botanical Sanctum": "GU",
    "Adarkar Wastes": "WU", "Underground River": "UB",
    "Sulfurous Springs": "BR", "Karplusan Forest": "RG",
    "Brushland": "GW", "Caves of Koilos": "WB",
    "Shivan Reef": "UR", "Llanowar Wastes": "BG",
    "Battlefield Forge": "RW", "Yavimaya Coast": "GU",
}


#  Helpers
def categorize_type(card_type: str) -> str:
    if pd.isna(card_type):
        return "other"
    t = card_type.lower()
    if "land" in t:        return "land"
    if "creature" in t:    return "creature"
    if "instant" in t:     return "instant"
    if "sorcery" in t:     return "sorcery"
    if "enchantment" in t: return "enchantment"
    if "artifact" in t:    return "artifact"
    if "planeswalker" in t: return "planeswalker"
    return "other"


def fuzzy_find(query, valid, n=5):
    return get_close_matches(query.lower(), [v.lower() for v in valid], n=n, cutoff=0.4)


def resolve_name(query, valid, label="card"):
    lower_map = {v.lower(): v for v in valid}
    q = query.strip().lower()
    if q in lower_map:
        return lower_map[q]
    matches = fuzzy_find(query, valid)
    if not matches:
        print(f" No match found for '{query}' in local data.")
        return None
    if len(matches) == 1:
        real = lower_map[matches[0]]
        print(f"  -> Matched '{query}' to: {real}")
        return real
    print(f"  Did you mean one of these {label}s?")
    for i, m in enumerate(matches, 1):
        print(f"    {i}. {lower_map[m]}")
    print(f"    0. None of these")
    while True:
        choice = input("  Pick a number: ").strip()
        if choice == "0":
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(matches):
                return lower_map[matches[idx]]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


#  Scryfall API client
class ScryfallClient:
    """
    Fetches card metadata from the Scryfall API.
    Caches results to JSON 
    Falls back to land-based color inference offline.
    """

    def __init__(self, cache_path=SCRYFALL_CACHE_FILE):
        self.cache_path = cache_path
        self.cache = {}
        self._api_available = None
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    self.cache = json.load(f)
                print(f"  Loaded Scryfall cache ({len(self.cache):,} cards)")
            except Exception:
                self.cache = {}

    def save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)

    def _check_api(self):
        if self._api_available is not None:
            return self._api_available
        try:
            import requests
            r = requests.get(f"{SCRYFALL_BASE}/cards/named",
                             params={"exact": "Sol Ring"}, timeout=5)
            self._api_available = r.ok
        except Exception:
            self._api_available = False
        if self._api_available:
            print("  Scryfall API connected")
        else:
            print("  Scryfall API not available — using offline color inference")
        return self._api_available

    def lookup_card(self, name):
        key = name.lower()
        if key in self.cache:
            return self.cache[key]
        if not self._check_api():
            return None
        import requests
        try:
            r = requests.get(f"{SCRYFALL_BASE}/cards/named",
                             params={"fuzzy": name}, timeout=8)
            if not r.ok:
                return None
            entry = self._parse(r.json())
            self.cache[key] = entry
            time.sleep(0.1)
            return entry
        except Exception:
            return None

    def batch_lookup(self, names, progress=True):
        to_fetch = [n for n in names if n.lower() not in self.cache]
        if not to_fetch:
            return
        if not self._check_api():
            return
        import requests
        total = len(to_fetch)
        fetched = 0
        if progress:
            print(f"  Fetching {total:,} cards from Scryfall …")
        for i in range(0, total, 75):
            batch = to_fetch[i:i+75]
            try:
                r = requests.post(f"{SCRYFALL_BASE}/cards/collection",
                                  json={"identifiers": [{"name": n} for n in batch]},
                                  timeout=30)
                if r.ok:
                    for card in r.json().get("data", []):
                        entry = self._parse(card)
                        self.cache[entry["name"].lower()] = entry
                    fetched += len(r.json().get("data", []))
                    if progress and (i // 75) % 20 == 0:
                        print(f"    {min(100, int(fetched/total*100))}% "
                              f"({fetched:,}/{total:,})")
            except Exception as e:
                print(f"  Scryfall batch error: {e}")
            time.sleep(0.12)
        if progress:
            print(f"    Done,  {fetched:,} cards fetched")
        self.save_cache()

    @staticmethod
    def _parse(data):
        ci = data.get("color_identity", [])
        prices = data.get("prices", {})
        price = float(prices.get("usd") or prices.get("usd_foil") or 0)
        return {
            "name": data.get("name", ""),
            "color_identity": sorted(ci),
            "type_line": data.get("type_line", ""),
            "category": categorize_type(data.get("type_line", "")),
            "rarity": data.get("rarity", "unknown"),
            "price": price,
        }

    def get_color_identity(self, name):
        key = name.lower()
        if key in self.cache:
            return self.cache[key].get("color_identity")
        return None

    def infer_commander_colors(self, df):
        """Infer commander CI from known lands in their decks."""
        result = {}
        for cmdr, grp in df.groupby("Commander"):
            key = cmdr.lower()
            if key in self.cache:
                result[cmdr] = set(self.cache[key].get("color_identity", []))
                continue
            cards = set(grp["Card_Name"].unique())
            colors = set()
            for land, c in _LAND_COLOR_MAP.items():
                if land in cards:
                    colors.update(c)
            result[cmdr] = colors
        return result

    def infer_card_colors(self, df, cmdr_colors):
        """Infer card CI = intersection of all host commanders' CIs."""
        card_colors = {}
        for (cmdr, card), _ in df.groupby(["Commander", "Card_Name"]).size().items():
            c_ci = cmdr_colors.get(cmdr)
            if c_ci is None:
                continue
            if card not in card_colors:
                card_colors[card] = set(c_ci)
            else:
                card_colors[card] &= c_ci
        return card_colors


#  Core engine
class DeckBuilder:

    def __init__(self, csv_path):
        print("=" * 60)
        print("  MTG COMMANDER DECK BUILDER")
        print("  Collaborative Filtering")
        print("=" * 60)
        print()
        self.scryfall = ScryfallClient()
        self._load_data(csv_path)
        self._resolve_colors()
        self._build_similarity()

    def _load_data(self, csv_path):
        print("[1/4] Loading deck data …")
        self.df = pd.read_csv(csv_path)
        self.df["category"] = self.df["Card_Type"].apply(categorize_type)
        self.deck_ids = {url: i for i, url in enumerate(self.df["Deck_URL"].unique())}
        self.df["deck_id"] = self.df["Deck_URL"].map(self.deck_ids)

        card_info = (
            self.df.groupby("Card_Name")
            .agg(Card_Type=("Card_Type", "first"),
                 category=("category", "first"),
                 Rarity=("Rarity", "first"),
                 Price_USD=("Price_USD", "median"),
                 deck_count=("deck_id", "nunique"))
            .reset_index()
        )
        self.card_info = card_info.set_index("Card_Name")
        self.all_cards = list(self.card_info.index)
        self.commanders = sorted(self.df["Commander"].unique())
        self.commander_cards = {}
        for cmdr, grp in self.df.groupby("Commander"):
            self.commander_cards[cmdr] = set(grp["Card_Name"].unique())
        print(f"     {len(self.all_cards):,} cards | "
              f"{len(self.deck_ids):,} decks | "
              f"{len(self.commanders)} commanders")

    def _resolve_colors(self):
        print("[2/4] Resolving color identities …")
        # Try Scryfall for all cards
        self.scryfall.batch_lookup(self.all_cards, progress=True)

        # Commander CIs
        self.commander_ci = self.scryfall.infer_commander_colors(self.df)

        # Card CIs — prefer Scryfall, fall back to inference
        self.card_ci = {}
        miss = 0
        for card in self.all_cards:
            ci = self.scryfall.get_color_identity(card)
            if ci is not None:
                self.card_ci[card] = set(ci)
            else:
                miss += 1
        if miss > 0:
            inferred = self.scryfall.infer_card_colors(self.df, self.commander_ci)
            for card, colors in inferred.items():
                if card not in self.card_ci:
                    self.card_ci[card] = colors
        print(f"     {len(self.all_cards)-miss:,} via Scryfall | "
              f"{miss:,} inferred from deck data")

    def _build_similarity(self):
        print("[3/4] Building co-occurrence matrix …")
        card_to_idx = {c: i for i, c in enumerate(self.all_cards)}
        n_cards = len(self.all_cards)

        rows, cols = [], []
        for _, row in self.df.iterrows():
            rows.append(card_to_idx[row["Card_Name"]])
            cols.append(row["deck_id"])
        co = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n_cards, len(self.deck_ids)),
        )

        print("[4/4] Computing cosine similarity …")
        t0 = time.time()
        self.sim_matrix = cosine_similarity(co, dense_output=False)
        print(f"     Done in {time.time()-t0:.1f}s  ({n_cards}×{n_cards})")

        self.card_to_idx = card_to_idx
        self.idx_to_card = {i: c for c, i in card_to_idx.items()}
        self._prices = np.array([
            float(self.card_info.loc[self.idx_to_card[i], "Price_USD"])
            for i in range(n_cards)
        ])
        self._price_factor = 1.0 / (1.0 + np.log1p(self._prices) * 0.15)
        print()

    # ── color check ───────────────────────────
    def is_legal(self, card, commander):
        cmdr_ci = self.commander_ci.get(commander)
        card_ci = self.card_ci.get(card)
        if cmdr_ci is None or card_ci is None:
            return True
        return card_ci.issubset(cmdr_ci)

    def _fmt_ci(self, commander):
        ci = self.commander_ci.get(commander, set())
        if not ci:
            return "C"
        return "".join(c for c in "WUBRG" if c in ci)

    # ── scoring ───────────────────────────────
    def _score(self, cmdr_idx, seed_idx, exclude):
        """
        Weighted scoring: 80% commander affinity, 20% seed affinity.
        """
        n = len(self.all_cards)
        if cmdr_idx:
            cs = self.sim_matrix[cmdr_idx].toarray().mean(axis=0).flatten()
        else:
            cs = np.zeros(n)
        if seed_idx:
            ss = self.sim_matrix[seed_idx].toarray().mean(axis=0).flatten()
        else:
            ss = np.zeros(n)
        scores = COMMANDER_WEIGHT * cs + SEED_WEIGHT * ss
        for i in exclude:
            scores[i] = -1
        return scores

    # ── build deck ────────────────────────────
    def build_deck(self, commander, seed_cards, budget=None,
                   external_seeds=None):
        external_seeds = external_seeds or []
        ext_names = {e["Card_Name"] for e in external_seeds}

        slots = DECK_SIZE - 1
        selected = []
        selected_idx = set()
        spent = 0.0

        # Precompute illegal set for this commander
        illegal = {self.card_to_idx[c] for c in self.all_cards
                   if not self.is_legal(c, commander)}

        # Lock seeds that exist in the co-occurrence matrix
        for card in seed_cards:
            if card in self.card_to_idx:
                selected.append(card)
                selected_idx.add(self.card_to_idx[card])

        # Reserve slots for all seeds
        slots -= len(selected) + len(ext_names)

        # Commander index pool
        cmdr_pool = self.commander_cards.get(commander, set())
        cmdr_idx = [self.card_to_idx[c] for c in cmdr_pool
                    if c in self.card_to_idx]
        seed_idx = list(selected_idx)

        type_counts = Counter()
        for card in selected:
            type_counts[self.card_info.loc[card, "category"]] += 1

        ci_str = self._fmt_ci(commander)
        legal_count = len(self.all_cards) - len(illegal)
        print(f"\n  Commander CI: [{ci_str}]")
        print(f"  Legal cards: {legal_count:,} / {len(self.all_cards):,}")
        print(f"  Filling {slots} remaining slots …")

        budget_exhausted = False

        while slots > 0:
            scores = self._score(cmdr_idx, seed_idx, selected_idx)
            adjusted = scores * self._price_factor
            ranked = np.argsort(adjusted)[::-1]

            needs_min = {cat for cat, (mn, _) in TYPE_TARGETS.items()
                         if type_counts[cat] < mn}

            picked = False
            for idx in ranked:
                if scores[idx] <= 0:
                    break
                if idx in illegal:
                    continue

                card = self.idx_to_card[idx]
                info = self.card_info.loc[card]
                cat = info["category"]
                price = float(info["Price_USD"])

                _, mx = TYPE_TARGETS.get(cat, (0, 99))
                if type_counts[cat] >= mx:
                    continue

                # Force-fill needed categories when running low on slots
                if needs_min and cat not in needs_min and slots <= sum(
                    max(0, TYPE_TARGETS[c][0] - type_counts[c])
                    for c in needs_min
                ):
                    continue

                if budget is not None and not budget_exhausted:
                    if spent + price > budget:
                        continue

                selected.append(card)
                selected_idx.add(idx)
                seed_idx.append(idx)
                type_counts[cat] += 1
                spent += price
                slots -= 1
                picked = True
                break

            if not picked:
                if budget is not None and not budget_exhausted:
                    budget_exhausted = True
                    print(f"  Budget ${budget:,.2f} reached (${spent:,.2f}). "
                          "Filling with cheapest legal cards …")
                    continue
                else:
                    print(f"  Stuck at {DECK_SIZE-1-slots} cards + commander.")
                    break

        for cat, (mn, _) in TYPE_TARGETS.items():
            if type_counts[cat] < mn:
                print(f"  {cat}: {type_counts[cat]}/{mn} minimum")

        # Build result
        rows = []
        cmdr_price = (float(self.card_info.loc[commander, "Price_USD"])
                      if commander in self.card_info.index else 0.0)
        rows.append({
            "Card_Name": commander, "Category": "commander",
            "Card_Type": (self.card_info.loc[commander, "Card_Type"]
                          if commander in self.card_info.index else ""),
            "Rarity": (self.card_info.loc[commander, "Rarity"]
                       if commander in self.card_info.index else ""),
            "Price_USD": cmdr_price, "Source": "commander",
        })
        for card in selected:
            info = self.card_info.loc[card]
            rows.append({
                "Card_Name": card, "Category": info["category"],
                "Card_Type": info["Card_Type"], "Rarity": info["Rarity"],
                "Price_USD": info["Price_USD"],
                "Source": "seed" if card in seed_cards else "recommended",
            })
        # Append external seeds from Scryfall
        for ext in external_seeds:
            rows.append({
                "Card_Name": ext["Card_Name"], "Category": ext["Category"],
                "Card_Type": ext["Card_Type"], "Rarity": ext["Rarity"],
                "Price_USD": ext["Price_USD"], "Source": "seed",
            })
        return pd.DataFrame(rows)

    # ── display ───────────────────────────────
    @staticmethod
    def print_deck(deck):
        total = deck["Price_USD"].sum()
        rec_cost = deck.loc[deck["Source"] == "recommended", "Price_USD"].sum()
        print()
        print("═" * 60)
        print("  GENERATED DECKLIST")
        print("═" * 60)
        for cat in ["commander", "creature", "instant", "sorcery",
                     "artifact", "enchantment", "planeswalker", "land", "other"]:
            grp = deck[deck["Category"] == cat].sort_values("Card_Name")
            if grp.empty:
                continue
            hdr = cat.upper()
            print(f"\n── {hdr} ({len(grp)}) {'─' * (40 - len(hdr))}")
            for _, r in grp.iterrows():
                src = {" [seed]": r["Source"] == "seed",
                       " [cmdr]": r["Source"] == "commander"}.get(True, "")
                if r["Source"] == "seed":
                    src = " [seed]"
                elif r["Source"] == "commander":
                    src = " [cmdr]"
                else:
                    src = ""
                print(f"  {r['Card_Name']:<45s} ${r['Price_USD']:>7.2f}{src}")
        print()
        print("─" * 60)
        tc = deck[deck["Category"] != "commander"]["Category"].value_counts()
        print("  Type breakdown:  " +
              "  ".join(f"{k}: {v}" for k, v in sorted(tc.items())))
        print(f"  Total cards:       {len(deck)}")
        print(f"  Recommended cost:  ${rec_cost:,.2f}")
        print(f"  Total deck value:  ${total:,.2f}")
        print("─" * 60)

    def list_commanders(self, query=""):
        matches = [c for c in self.commanders if query.lower() in c.lower()]
        if not matches:
            print("No commanders matched.")
            return
        print(f"\nCommanders matching '{query}' ({len(matches)}):")
        for i, c in enumerate(matches, 1):
            ci = self._fmt_ci(c)
            n = len(self.commander_cards.get(c, []))
            print(f"  {i:3d}. {c:<40s}  [{ci:<5s}]  ({n} cards)")

    def list_popular_cards(self, n=30):
        top = self.card_info.nlargest(n, "deck_count")
        print(f"\nTop {n} most-played cards:")
        for i, (name, row) in enumerate(top.iterrows(), 1):
            ci = self.card_ci.get(name, set())
            ci_s = "".join(sorted(ci)) or "C"
            print(f"  {i:3d}. {name:<40s} [{ci_s:<5s}]  "
                  f"in {int(row['deck_count']):>4d} decks  "
                  f"${row['Price_USD']:.2f}")

    def export_deck(self, deck, path):
        deck.to_csv(path, index=False)
        print(f"\n  ✓ Saved to {path}")

    def lookup_unknown_card(self, name):
        entry = self.scryfall.lookup_card(name)
        if entry is None:
            return None
        return {
            "Card_Name": entry["name"],
            "Category": entry["category"],
            "Card_Type": entry["type_line"],
            "Rarity": entry["rarity"],
            "Price_USD": entry["price"],
            "color_identity": set(entry["color_identity"]),
        }


#  Interactive terminal session
def interactive_session(builder):
    print("\n" + "=" * 60)
    print("  INTERACTIVE DECK BUILDER")
    print("=" * 60)
    print("  Commands:  list [query]  — search commanders")
    print("             popular       — show popular cards")
    print("             build         — start building a deck")
    print("             quit          — exit")
    print("=" * 60)
    while True:
        print()
        cmd = input("deck-builder> ").strip()
        if not cmd:
            continue
        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        if action in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        elif action == "list":
            builder.list_commanders(parts[1] if len(parts) > 1 else "")
        elif action == "popular":
            builder.list_popular_cards()
        elif action == "build":
            _build_wizard(builder)
        else:
            print("Unknown command. Try: list, popular, build, quit")


def _build_wizard(builder):
    print("\n── DECK BUILDER WIZARD ──")

    # Commander
    while True:
        raw = input("\n  Commander name (or 'list' to search): ").strip()
        if raw.lower() == "list":
            builder.list_commanders(input("  Search query: ").strip())
            continue
        commander = resolve_name(raw, builder.commanders, label="commander")
        if commander:
            break
        print("  Please try again.")
    ci_str = builder._fmt_ci(commander)
    print(f"\n  ✓ Commander: {commander}  [{ci_str}]")

    # Seed cards
    print("\n  Enter cards you want in the deck (one per line).")
    print("  Cards must be legal in your commander's colors.")
    print("  Unknown cards will be looked up via Scryfall.")
    print("  Type 'done' when finished, or 'skip' for none.")

    seed_cards = []
    ext_seeds = []

    while True:
        raw = input("  Card: ").strip()
        if raw.lower() in ("done", "skip", ""):
            break

        card = resolve_name(raw, builder.all_cards, label="card")
        if card:
            if not builder.is_legal(card, commander):
                ci = builder.card_ci.get(card, set())
                print(f"    {card} [{','.join(sorted(ci)) or 'C'}] is "
                      f"not legal in a [{ci_str}] deck.")
                continue
            if card not in seed_cards:
                seed_cards.append(card)
                print(f"    Added: {card}")
            else:
                print(f"    (already added)")
        else:
            print(f"  Searching Scryfall for '{raw}' …")
            result = builder.lookup_unknown_card(raw)
            if result is None:
                print(f"    Not found. (Scryfall may be unreachable, "
                      f"works when run locally)")
                continue
            cmdr_ci = builder.commander_ci.get(commander, set())
            if not result["color_identity"].issubset(cmdr_ci):
                ci_s = "".join(sorted(result["color_identity"])) or "C"
                print(f"    {result['Card_Name']} [{ci_s}] not "
                      f"legal in [{ci_str}].")
                continue
            name = result["Card_Name"]
            if name not in seed_cards:
                ext_seeds.append(result)
                seed_cards.append(name)
                print(f"    Added (Scryfall): {name}  "
                      f"[{result['Category']}] ${result['Price_USD']:.2f}")

    print(f"\n  Seeds: {len(seed_cards)} "
          f"({len(ext_seeds)} via Scryfall)")

    # Budget
    braw = input("\n  Max budget USD (Enter = no limit): ").strip()
    budget = None
    if braw:
        try:
            budget = float(braw.replace("$", "").replace(",", ""))
            print(f"   Budget: ${budget:,.2f}")
        except ValueError:
            print("  Could not parse, no limit.")

    # Build
    print("\n  Building deck …")
    deck = builder.build_deck(commander, seed_cards, budget,
                              external_seeds=ext_seeds)

    builder.print_deck(deck)

    if input("\n  Save to CSV? (y/n): ").strip().lower() == "y":
        fname = input("  Filename [my_deck.csv]: ").strip() or "my_deck.csv"
        builder.export_deck(deck, fname)


#  Demo mode (ignore for final version :P) 
def demo_build(builder):
    print("\n" + "=" * 60)
    print("  DEMO BUILD")
    print("=" * 60)
    cmdr = "Krenko Tin Street Kingpin"
    seeds = ["Sol Ring", "Lightning Bolt"]
    print(f"  Commander:  {cmdr}  [{builder._fmt_ci(cmdr)}]")
    print(f"  Seeds:      {seeds}")
    print(f"  Budget:     No limit")
    deck = builder.build_deck(cmdr, seeds, budget=None)
    builder.print_deck(deck)
    return deck


def main():
    csv_path = None
    for p in ["mtg_commander_data.csv",
              "/mnt/user-data/uploads/mtg_commander_data.csv",
              os.path.join(os.path.dirname(__file__), "mtg_commander_data.csv")]:
        if os.path.exists(p):
            csv_path = p
            break
    if not csv_path:
        print("ERROR: Cannot find mtg_commander_data.csv")
        sys.exit(1)

    builder = DeckBuilder(csv_path)
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_build(builder)
    else:
        interactive_session(builder)


if __name__ == "__main__":
    main()