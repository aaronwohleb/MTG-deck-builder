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
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy import sparse
from collections import Counter
from difflib import get_close_matches
import re

#  Constants
DECK_SIZE = 100
SCRYFALL_CACHE_FILE = "scryfall_cache.json"
SCRYFALL_BASE = "https://api.scryfall.com"

BASIC_LANDS = {
    "Plains", "Island", "Swamp", "Mountain", "Forest",
    "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
    "Snow-Covered Mountain", "Snow-Covered Forest",
    "Wastes",
}

# Max total basic lands in a deck; distributed evenly across commander colors
BASIC_LAND_MAX = 25

# Maps each basic land to its color identity letter
BASIC_LAND_COLORS = {
    "Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G",
    "Snow-Covered Plains": "W", "Snow-Covered Island": "U",
    "Snow-Covered Swamp": "B", "Snow-Covered Mountain": "R",
    "Snow-Covered Forest": "G",
    "Wastes": "C",
}

# Commander pool gets 60% of the scoring weight
W_CMDR = 0.25
W_SEED = 0.15
W_ARCH = 0.60

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

# Target mana curve for non-land spells (proportion of ~60 spell slots)
# Index = CMC bucket;  0,1,2,3,4,5,6,7+
CURVE_TARGETS = [0.05, 0.12, 0.22, 0.20, 0.15, 0.11, 0.08, 0.07]
CURVE_PENALTY_STRENGTH = 0.25  # how hard to penalize over-represented CMC
 
# Archetype clustering
N_ARCHETYPES = 6
SVD_DIMS = 20
IDF_STRENGTH = 0.3  # 0 = disabled, 1 = full boost/penalty for card frequency

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


def count_pips(mana_cost):
    """Return a Counter of colored pips in a Scryfall mana cost string, e.g. '{2}{U}{U}'."""
    pips = Counter()
    for symbol in re.findall(r'\{([^}]+)\}', mana_cost or ""):
        for color in "WUBRG":
            if color in symbol:
                pips[color] += 1
    return pips


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
            "mana_cost": data.get("mana_cost", ""),
        }

    def get_color_identity(self, name):
        key = name.lower()
        if key in self.cache:
            return self.cache[key].get("color_identity")
        return None

    def get_cmc(self, name):
        key = name.lower()
        if key in self.cache:
            return self.cache[key].get("cmc")
        return None

    def get_mana_cost(self, name):
        key = name.lower()
        if key in self.cache:
            return self.cache[key].get("mana_cost") or ""
        return ""

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
    

#  Archetype Clusterer (Fuzzy)
class ArchetypeClusterer:
    """
    Clusters all decks into K strategic archetypes using:
      1. TruncatedSVD to reduce the deck×card matrix
      2. KMeans to find cluster centroids
      3. Softmax of inverse distances → fuzzy membership
 
    Each deck gets a probability vector over archetypes.
    Clusters are auto-labeled by their most distinctive cards.
    """
 
    def __init__(self, deck_card_matrix, all_cards, n_clusters=N_ARCHETYPES,
                 svd_dims=SVD_DIMS):
        self.n_clusters = n_clusters
        self.all_cards = all_cards
 
        print(f"  Reducing {deck_card_matrix.shape[1]:,} dimensions "
              f"→ {svd_dims} with SVD …")
        self.svd = TruncatedSVD(n_components=svd_dims, random_state=42)
        self.deck_vecs = self.svd.fit_transform(deck_card_matrix)
        var = self.svd.explained_variance_ratio_.sum()
        print(f"  Explained variance: {var:.1%}")
 
        print(f"  Clustering into {n_clusters} archetypes (KMeans) …")
        self.km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.hard_labels = self.km.fit_predict(self.deck_vecs)
 
        # Compute soft membership via softmax of negative distance
        self.membership = self._soft_membership(self.deck_vecs)
 
        # Auto-label each cluster
        self.labels = self._auto_label(deck_card_matrix)
 
    def _soft_membership(self, vecs):
        """
        Fuzzy membership: for each deck, compute probability of
        belonging to each cluster using softmax(−distance²).
        """
        dists = np.zeros((len(vecs), self.n_clusters))
        for k in range(self.n_clusters):
            diff = vecs - self.km.cluster_centers_[k]
            dists[:, k] = np.sum(diff ** 2, axis=1)
        # Softmax of negative distances (temperature=1)
        neg = -dists
        neg -= neg.max(axis=1, keepdims=True)  # numerical stability
        exp = np.exp(neg)
        return exp / exp.sum(axis=1, keepdims=True)
 
    def _auto_label(self, deck_card_matrix):
        """
        Name each cluster by its most distinctive card types/themes.
        Picks the cards with the highest frequency ratio vs overall.
        """
        overall_freq = np.array(deck_card_matrix.mean(axis=0)).flatten()
        labels = []
        for k in range(self.n_clusters):
            mask = self.hard_labels == k
            cluster_freq = np.array(
                deck_card_matrix[mask].mean(axis=0)
            ).flatten()
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(overall_freq > 0.005,
                                 cluster_freq / overall_freq, 0)
            top_idx = np.argsort(ratio)[::-1][:5]
            top_names = [self.all_cards[i] for i in top_idx]
            labels.append(top_names)
        return labels
 
    def get_archetype_deck_indices(self, k):
        """Return deck indices that strongly belong to archetype k."""
        return np.where(self.membership[:, k] > 0.3)[0]
 
    def print_archetypes(self):
        print(f"\n  Available Archetypes ({self.n_clusters} clusters):")
        for k in range(self.n_clusters):
            n = (self.hard_labels == k).sum()
            top = ", ".join(self.labels[k][:3])
            print(f"    {k+1}. [{n:>3d} decks]  {top}")
 
    def commander_archetypes(self, deck_ids_for_cmdr):
        """
        Given a list of deck indices belonging to a commander,
        return the average membership vector and dominant archetype.
        """
        if not deck_ids_for_cmdr:
            return np.ones(self.n_clusters) / self.n_clusters, 0
        mem = self.membership[deck_ids_for_cmdr].mean(axis=0)
        return mem, int(np.argmax(mem))
 
 
#  Mana Curve Optimizer
class ManaCurve:
    """
    Tracks the mana cost distribution of non-land spells and
    penalizes candidates at over-represented CMC values.
 
    CMC buckets: 0, 1, 2, 3, 4, 5, 6, 7+
    """
 
    def __init__(self, targets=None, strength=CURVE_PENALTY_STRENGTH):
        self.targets = targets or CURVE_TARGETS
        self.strength = strength
        self.counts = np.zeros(8)
        self.total = 0
 
    def cmc_bucket(self, cmc):
        return min(int(cmc), 7)
 
    def add(self, cmc):
        self.counts[self.cmc_bucket(cmc)] += 1
        self.total += 1
 
    def penalty(self, cmc):
        """
        Returns a multiplier in (0, 1] that penalizes candidates
        whose CMC bucket is already above the target proportion.
 
        If bucket is at or below target → 1.0 (no penalty)
        If bucket is over target → decays toward (1 − strength)
        """
        if self.total == 0:
            return 1.0
        bucket = self.cmc_bucket(cmc)
        actual_prop = self.counts[bucket] / max(self.total, 1)
        target_prop = self.targets[bucket]
        if actual_prop <= target_prop:
            return 1.0
        # How far over target (0 = at target, 1 = 2× target)
        overshoot = (actual_prop - target_prop) / max(target_prop, 0.01)
        # Exponential decay
        return max(1.0 - self.strength * overshoot, 0.3)
 
    def histogram_str(self, total_spell_slots=60):
        """Format the current curve as an ASCII histogram."""
        lines = []
        max_bar = 20
        for i in range(8):
            label = f"{i}" if i < 7 else "7+"
            count = int(self.counts[i])
            target = self.targets[i] * total_spell_slots
            bar_len = int(count / max(self.counts.max(), 1) * max_bar)
            bar = "█" * bar_len
            marker = f" (target ~{target:.0f})"
            lines.append(f"    {label:>2s}│ {bar:<{max_bar}s} {count:>3d}{marker}")
        return "\n".join(lines)
 

#  Core engine
class DeckBuilder:

    def __init__(self, csv_path):
        print("=" * 60)
        print("  MTG COMMANDER DECK BUILDER")
        print("  Collab Filtering + Mana Curve + Archetypes")
        print("=" * 60)
        print()
        self.scryfall = ScryfallClient()
        self._load_data(csv_path)
        self._resolve_colors()
        self._resolve_cmc()
        self._build_similarity()
        self._build_archetypes()
 
    # ── data loading ──────────────────────────
    def _load_data(self, csv_path):
        print("[1/6] Loading deck data …")
        self.df = pd.read_csv(csv_path)
        self.df["category"] = self.df["Card_Type"].apply(categorize_type)
        self.deck_ids = {url: i for i, url in enumerate(self.df["Deck_URL"].unique())}
        self.df["deck_id"] = self.df["Deck_URL"].map(self.deck_ids)
        self.id_to_deck = {i: url for url, i in self.deck_ids.items()}
 
        card_info = (
            self.df.groupby("Card_Name")
            .agg(Card_Type=("Card_Type","first"),
                 category=("category","first"),
                 Rarity=("Rarity","first"),
                 Price_USD=("Price_USD","median"),
                 deck_count=("deck_id","nunique"))
            .reset_index()
        )
        self.card_info = card_info.set_index("Card_Name")
        self.all_cards = list(self.card_info.index)
        self.commanders = sorted(self.df["Commander"].unique())
        self.commander_cards = {}
        self.commander_deck_ids = {}
        for cmdr, grp in self.df.groupby("Commander"):
            self.commander_cards[cmdr] = set(grp["Card_Name"].unique())
            self.commander_deck_ids[cmdr] = list(grp["deck_id"].unique())
        print(f"     {len(self.all_cards):,} cards | "
              f"{len(self.deck_ids):,} decks | "
              f"{len(self.commanders)} commanders")
 
    # ── color identity ────────────────────────
    def _resolve_colors(self):
        print("[2/6] Resolving color identities …")
        self.scryfall.batch_lookup(self.all_cards, progress=True)
        self.commander_ci = self.scryfall.infer_commander_colors(self.df)
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
              f"{miss:,} inferred")
 
    # ── mana costs ────────────────────────────
    def _resolve_cmc(self):
        print("[3/6] Resolving mana costs …")
        self.card_cmc = {}
        found = 0
        for card in self.all_cards:
            cmc = self.scryfall.get_cmc(card)
            if cmc is not None:
                self.card_cmc[card] = cmc
                found += 1
            else:
                self.card_cmc[card] = None
        print(f"     {found:,} cards with CMC data "
              f"({len(self.all_cards)-found:,} unknown)")
 
    # ── similarity matrix ─────────────────────
    def _build_similarity(self):
        print("[4/6] Building co-occurrence matrix …")
        card_to_idx = {c: i for i, c in enumerate(self.all_cards)}
        n_cards = len(self.all_cards)
 
        rows, cols = [], []
        for _, row in self.df.iterrows():
            rows.append(card_to_idx[row["Card_Name"]])
            cols.append(row["deck_id"])
 
        # card × deck  (for cosine similarity between cards)
        self.co_card_deck = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n_cards, len(self.deck_ids)),
        )
        # deck × card  (for archetype clustering)
        self.co_deck_card = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (cols, rows)),
            shape=(len(self.deck_ids), n_cards),
        )
 
        print("[5/6] Computing cosine similarity …")
        t0 = time.time()
        self.sim_matrix = cosine_similarity(self.co_card_deck, dense_output=False)
        print(f"     Done in {time.time()-t0:.1f}s  ({n_cards}×{n_cards})")
 
        self.card_to_idx = card_to_idx
        self.idx_to_card = {i: c for c, i in card_to_idx.items()}
        self._prices = np.array([
            float(self.card_info.loc[self.idx_to_card[i], "Price_USD"])
            for i in range(n_cards)
        ])
        self._price_factor = 1.0 / (1.0 + np.log1p(self._prices) * 1.0)
 
        # Precompute CMC array for fast lookup
        self._cmc = np.array([
            self.card_cmc.get(self.idx_to_card[i], 0) or 0
            for i in range(n_cards)
        ])

        # IDF factor: boost cards appearing in few decks, penalize cards in many decks
        total_decks = len(self.deck_ids)
        raw_idf = np.array([
            np.log(total_decks / max(1, int(self.card_info.loc[self.idx_to_card[i], "deck_count"])))
            for i in range(n_cards)
        ])
        idf_min, idf_max = raw_idf.min(), raw_idf.max()
        idf_range = idf_max - idf_min
        idf_norm = (raw_idf - idf_min) / idf_range if idf_range > 0 else np.ones(n_cards)
        self._idf_factor = (1.0 - IDF_STRENGTH) + IDF_STRENGTH * idf_norm
        basic_mask = np.array([self.idx_to_card[i] in BASIC_LANDS for i in range(n_cards)])
        self._idf_factor[basic_mask] = 1.0

    # ── archetype clustering ──────────────────
    def _build_archetypes(self):
        print("[6/6] Clustering deck archetypes …")
        self.clusterer = ArchetypeClusterer(
            self.co_deck_card, self.all_cards,
            n_clusters=N_ARCHETYPES, svd_dims=SVD_DIMS
        )
        print()
 
    # ── helpers ───────────────────────────────
    def is_legal(self, card, commander):
        cmdr_ci = self.commander_ci.get(commander)
        card_ci = self.card_ci.get(card)
        if cmdr_ci is None or card_ci is None:
            return True
        return card_ci.issubset(cmdr_ci)
 
    def _fmt_ci(self, commander):
        ci = self.commander_ci.get(commander, set())
        if not ci: return "C"
        return "".join(c for c in "WUBRG" if c in ci)
 
    # ── scoring ───────────────────────────────
    def _score(self, cmdr_idx, seed_idx, arch_idx, exclude):
        """
        Three-signal weighted scoring:
          W_CMDR × mean(sim to commander pool)
        + W_SEED × mean(sim to growing seed pool)
        + W_ARCH × mean(sim to archetype's card pool)
        """
        n = len(self.all_cards)
        cs = (self.sim_matrix[cmdr_idx].toarray().mean(axis=0).flatten()
              if cmdr_idx else np.zeros(n))
        ss = (self.sim_matrix[seed_idx].toarray().mean(axis=0).flatten()
              if seed_idx else np.zeros(n))
        ar = (self.sim_matrix[arch_idx].toarray().mean(axis=0).flatten()
              if arch_idx else np.zeros(n))
 
        scores = W_CMDR * cs + W_SEED * ss + W_ARCH * ar
        for i in exclude:
            scores[i] = -1
        return scores
 
    # ── build deck ────────────────────────────
    def build_deck(self, commander, seed_cards, budget=None,
                   external_seeds=None, archetype=None):
        external_seeds = external_seeds or []
        ext_names = {e["Card_Name"] for e in external_seeds}
        slots = DECK_SIZE - 1
        selected = []
        selected_idx = set()   # similarity seed pool
        exclude_idx = set()    # non-basic cards ineligible for future picks
        spent = 0.0
 
        # Illegal cards for this commander
        illegal = {self.card_to_idx[c] for c in self.all_cards
                   if not self.is_legal(c, commander)}
 
        # Lock local seeds
        for card in seed_cards:
            if card in self.card_to_idx:
                selected.append(card)
                idx = self.card_to_idx[card]
                selected_idx.add(idx)
                if card not in BASIC_LANDS:
                    exclude_idx.add(idx)
        slots -= len(selected) + len(ext_names)
 
        # Commander index pool
        cmdr_pool = self.commander_cards.get(commander, set())
        cmdr_idx = [self.card_to_idx[c] for c in cmdr_pool
                    if c in self.card_to_idx]
        seed_idx = list(selected_idx)
 
        # Archetype index pool — cards frequent in chosen archetype's decks
        arch_idx = []
        if archetype is not None and 0 <= archetype < self.clusterer.n_clusters:
            arch_deck_ids = self.clusterer.get_archetype_deck_indices(archetype)
            if len(arch_deck_ids) > 0:
                arch_card_freq = np.array(
                    self.co_deck_card[arch_deck_ids].mean(axis=0)
                ).flatten()
                # Take top ~200 most frequent cards in this archetype
                top_arch = np.argsort(arch_card_freq)[::-1][:200]
                arch_idx = [int(i) for i in top_arch if arch_card_freq[i] > 0.05]
 
        # Basic land balance tracking
        basic_color_counts = Counter()
        pip_counts = Counter()  # colored pips across all selected non-land spells

        # Type + curve tracking
        type_counts = Counter()
        curve = ManaCurve()
        for card in selected:
            cat = self.card_info.loc[card, "category"]
            type_counts[cat] += 1
            if card in BASIC_LANDS:
                basic_color_counts[BASIC_LAND_COLORS.get(card, "C")] += 1
            elif cat != "land":
                mc = self.scryfall.get_mana_cost(card)
                if mc:
                    pip_counts += count_pips(mc)
                else:
                    for color in self.card_ci.get(card, set()):
                        pip_counts[color] += 1
            cmc = self.card_cmc.get(card)
            if cmc is not None and cat != "land":
                curve.add(cmc)
 
        ci_str = self._fmt_ci(commander)
        legal = len(self.all_cards) - len(illegal)
        has_cmc = sum(1 for c in self.all_cards if self.card_cmc.get(c) is not None)
        print(f"\n  Commander CI:    [{ci_str}]")
        print(f"  Legal cards:     {legal:,} / {len(self.all_cards):,}")
        if archetype is not None:
            arch_name = ", ".join(self.clusterer.labels[archetype][:3])
            print(f"  Archetype:       #{archetype+1} ({arch_name})")
        print(f"  CMC data:        {has_cmc:,} cards")
        print(f"  Filling {slots} remaining slots …")
 
        budget_exhausted = False
 
        while slots > 0:
            scores = self._score(cmdr_idx, seed_idx, arch_idx, exclude_idx)
 
            # Apply price and IDF penalties
            adjusted = scores * self._price_factor * self._idf_factor
 
            # Apply mana curve penalty
            for i in range(len(self.all_cards)):
                if adjusted[i] <= 0:
                    continue
                card_name = self.idx_to_card[i]
                cat = self.card_info.loc[card_name, "category"]
                if cat != "land":
                    cmc = self._cmc[i]
                    adjusted[i] *= curve.penalty(cmc)
 
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
                if card in BASIC_LANDS:
                    land_color = BASIC_LAND_COLORS.get(card, "C")
                    if sum(basic_color_counts.values()) >= BASIC_LAND_MAX:
                        continue
                    total_pips = sum(pip_counts.values())
                    if total_pips > 0:
                        pip_frac = pip_counts.get(land_color, 0) / total_pips
                        color_target = max(1, round(BASIC_LAND_MAX * pip_frac))
                        if basic_color_counts[land_color] >= color_target:
                            continue
                if needs_min and cat not in needs_min and slots <= sum(
                    max(0, TYPE_TARGETS[c][0] - type_counts[c])
                    for c in needs_min
                ):
                    continue
                if budget is not None and not budget_exhausted:
                    if spent + price > budget:
                        continue
 
                selected.append(card)
                if idx not in selected_idx:
                    selected_idx.add(idx)
                    seed_idx.append(idx)
                if card not in BASIC_LANDS:
                    exclude_idx.add(idx)
                else:
                    basic_color_counts[BASIC_LAND_COLORS.get(card, "C")] += 1
                type_counts[cat] += 1
                spent += price
                slots -= 1
                picked = True
 
                # Track mana curve and pip counts (non-lands only)
                if cat != "land":
                    cmc = self.card_cmc.get(card)
                    if cmc is not None:
                        curve.add(cmc)
                    mc = self.scryfall.get_mana_cost(card)
                    if mc:
                        pip_counts += count_pips(mc)
                    else:
                        for color in self.card_ci.get(card, set()):
                            pip_counts[color] += 1
                break
 
            if not picked:
                if budget is not None and not budget_exhausted:
                    budget_exhausted = True
                    print(f"  ⚠ Budget ${budget:,.2f} reached (${spent:,.2f}).")
                    continue
                else:
                    print(f"  ⚠ Stuck at {DECK_SIZE-1-slots} cards + commander.")
                    break
 
        for cat, (mn, _) in TYPE_TARGETS.items():
            if type_counts[cat] < mn:
                print(f"  ⚠ {cat}: {type_counts[cat]}/{mn} minimum")
 
        # Build result
        rows = []
        cmdr_price = (float(self.card_info.loc[commander, "Price_USD"])
                      if commander in self.card_info.index else 0.0)
        cmdr_cmc = self.card_cmc.get(commander, 0) or 0
        rows.append({
            "Card_Name": commander, "Category": "commander",
            "Card_Type": (self.card_info.loc[commander, "Card_Type"]
                          if commander in self.card_info.index else ""),
            "Rarity": (self.card_info.loc[commander, "Rarity"]
                       if commander in self.card_info.index else ""),
            "Price_USD": cmdr_price, "CMC": cmdr_cmc, "Source": "commander",
        })
        for card in selected:
            info = self.card_info.loc[card]
            rows.append({
                "Card_Name": card, "Category": info["category"],
                "Card_Type": info["Card_Type"], "Rarity": info["Rarity"],
                "Price_USD": info["Price_USD"],
                "CMC": self.card_cmc.get(card, 0) or 0,
                "Source": "seed" if card in seed_cards else "recommended",
            })
        for ext in external_seeds:
            rows.append({
                "Card_Name": ext["Card_Name"], "Category": ext["Category"],
                "Card_Type": ext["Card_Type"], "Rarity": ext["Rarity"],
                "Price_USD": ext["Price_USD"],
                "CMC": ext.get("CMC", 0), "Source": "seed",
            })
 
        self._last_curve = curve
        return pd.DataFrame(rows)
 
    # ── display ───────────────────────────────
    def print_deck(self, deck):
        total = deck["Price_USD"].sum()
        rec_cost = deck.loc[deck["Source"] == "recommended", "Price_USD"].sum()
        print()
        print("═" * 60)
        print("  GENERATED DECKLIST")
        print("═" * 60)
        for cat in ["commander", "creature", "instant", "sorcery",
                     "artifact", "enchantment", "planeswalker", "land", "other"]:
            grp = deck[deck["Category"] == cat].sort_values("Card_Name")
            if grp.empty: continue
            hdr = cat.upper()
            print(f"\n── {hdr} ({len(grp)}) {'─' * (40 - len(hdr))}")
            for _, r in grp.iterrows():
                if r["Source"] == "seed":       src = " [seed]"
                elif r["Source"] == "commander": src = " [cmdr]"
                else:                           src = ""
                cmc_val = r.get('CMC')
                has_cmc = cmc_val is not None and cmc_val > 0
                cmc_s = f"({int(cmc_val)})" if has_cmc else ""
                if cat in ("commander", "land"):
                    cmc_s = ""
                print(f"  {r['Card_Name']:<42s} {cmc_s:>4s}"
                      f" ${r['Price_USD']:>7.2f}{src}")
        print()
        print("─" * 60)
        tc = deck[deck["Category"] != "commander"]["Category"].value_counts()
        print("  Type breakdown:  " +
              "  ".join(f"{k}: {v}" for k, v in sorted(tc.items())))
 
        # Mana curve display
        if hasattr(self, '_last_curve') and self._last_curve.total > 1:
            non_land = len(deck[~deck["Category"].isin(["commander", "land"])])
            print(f"\n  Mana Curve (non-land spells):")
            print(self._last_curve.histogram_str(non_land))
        elif hasattr(self, '_last_curve') and self._last_curve.total == 0:
            print(f"\n  Mana Curve: no CMC data (connect to Scryfall to enable)")
 
        print(f"\n  Total cards:       {len(deck)}")
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
            dids = self.commander_deck_ids.get(c, [])
            _, dom = self.clusterer.commander_archetypes(dids)
            arch_label = ", ".join(self.clusterer.labels[dom][:2])
            n = len(self.commander_cards.get(c, []))
            print(f"  {i:3d}. {c:<35s} [{ci:<5s}]  "
                  f"({n} cards) arch: {arch_label}")
 
    def list_popular_cards(self, n=30):
        top = self.card_info.nlargest(n, "deck_count")
        print(f"\nTop {n} most-played cards:")
        for i, (name, row) in enumerate(top.iterrows(), 1):
            ci = self.card_ci.get(name, set())
            ci_s = "".join(sorted(ci)) or "C"
            cmc = self.card_cmc.get(name)
            cmc_s = f"CMC {int(cmc)}" if cmc is not None else ""
            print(f"  {i:3d}. {name:<35s} [{ci_s:<5s}] {cmc_s:>5s}  "
                  f"in {int(row['deck_count']):>4d} decks  "
                  f"${row['Price_USD']:.2f}")
 
    def export_deck(self, deck, path):
        deck.to_csv(path, index=False)
        print(f"\n  ✓ Saved to {path}")
 
    def lookup_unknown_card(self, name):
        entry = self.scryfall.lookup_card(name)
        if entry is None: return None
        return {
            "Card_Name": entry["name"],
            "Category": entry["category"],
            "Card_Type": entry["type_line"],
            "Rarity": entry["rarity"],
            "Price_USD": entry["price"],
            "CMC": entry.get("cmc", 0),
            "color_identity": set(entry["color_identity"]),
        }
 
 
# ─────────────────────────────────────────────
#  Interactive CLI
# ─────────────────────────────────────────────
def interactive_session(builder):
    print("\n" + "=" * 60)
    print("  INTERACTIVE DECK BUILDER")
    print("=" * 60)
    print("  Commands:  list [query]  — search commanders")
    print("             popular       — show popular cards")
    print("             archetypes    — show deck archetypes")
    print("             build         — start building a deck")
    print("             quit          — exit")
    print("=" * 60)
    while True:
        print()
        cmd = input("deck-builder> ").strip()
        if not cmd: continue
        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        if action in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        elif action == "list":
            builder.list_commanders(parts[1] if len(parts) > 1 else "")
        elif action == "popular":
            builder.list_popular_cards()
        elif action in ("archetypes", "arch"):
            builder.clusterer.print_archetypes()
        elif action == "build":
            _build_wizard(builder)
        else:
            print("Unknown command. Try: list, popular, archetypes, build, quit")
 
 
def _build_wizard(builder):
    print("\n── DECK BUILDER WIZARD ──")
 
    # 1. Commander
    while True:
        raw = input("\n  Commander name (or 'list' to search): ").strip()
        if raw.lower() == "list":
            builder.list_commanders(input("  Search query: ").strip())
            continue
        commander = resolve_name(raw, builder.commanders, label="commander")
        if commander: break
        print("  Please try again.")
    ci_str = builder._fmt_ci(commander)
    print(f"\n  ✓ Commander: {commander}  [{ci_str}]")
 
    # 2. Archetype selection
    dids = builder.commander_deck_ids.get(commander, [])
    mem, dominant = builder.clusterer.commander_archetypes(dids)
 
    print(f"\n  Archetype affinity for {commander}:")
    for k in range(builder.clusterer.n_clusters):
        label = ", ".join(builder.clusterer.labels[k][:3])
        bar_len = int(mem[k] * 30)
        bar = "█" * bar_len
        dom_marker = " ◄" if k == dominant else ""
        print(f"    {k+1}. {bar:<30s} {mem[k]:>5.1%}  {label}{dom_marker}")
 
    arch_raw = input(f"\n  Choose archetype (1-{builder.clusterer.n_clusters}), "
                     f"Enter for default [{dominant+1}]: ").strip()
    if arch_raw:
        try:
            archetype = int(arch_raw) - 1
            if not (0 <= archetype < builder.clusterer.n_clusters):
                archetype = dominant
        except ValueError:
            archetype = dominant
    else:
        archetype = dominant
    arch_label = ", ".join(builder.clusterer.labels[archetype][:3])
    print(f"  ✓ Archetype: #{archetype+1} ({arch_label})")
 
    # 3. Seed cards
    print("\n  Enter cards you want in the deck (one per line).")
    print("  Cards must be legal in your commander's colors.")
    print("  Unknown cards will be looked up via Scryfall.")
    print("  Type 'done' when finished, or 'skip' for none.")
 
    seed_cards = []
    ext_seeds = []
 
    while True:
        raw = input("  Card: ").strip()
        if raw.lower() in ("done", "skip", ""): break
        card = resolve_name(raw, builder.all_cards, label="card")
        if card:
            if not builder.is_legal(card, commander):
                ci = builder.card_ci.get(card, set())
                print(f"    ✗ {card} [{','.join(sorted(ci)) or 'C'}] is "
                      f"not legal in a [{ci_str}] deck.")
                continue
            if card not in seed_cards:
                seed_cards.append(card)
                print(f"    ✓ Added: {card}")
            else:
                print(f"    (already added)")
        else:
            print(f"  Searching Scryfall for '{raw}' …")
            result = builder.lookup_unknown_card(raw)
            if result is None:
                print(f"    ✗ Not found. (Scryfall works when run locally)")
                continue
            cmdr_ci = builder.commander_ci.get(commander, set())
            if not result["color_identity"].issubset(cmdr_ci):
                ci_s = "".join(sorted(result["color_identity"])) or "C"
                print(f"    ✗ {result['Card_Name']} [{ci_s}] not "
                      f"legal in [{ci_str}].")
                continue
            name = result["Card_Name"]
            if name not in seed_cards:
                ext_seeds.append(result)
                seed_cards.append(name)
                print(f"    ✓ Added (Scryfall): {name}  "
                      f"[{result['Category']}] ${result['Price_USD']:.2f}")
 
    print(f"\n  Seeds: {len(seed_cards)} "
          f"({len(ext_seeds)} via Scryfall)")
 
    # 4. Budget
    braw = input("\n  Max budget USD (Enter = no limit): ").strip()
    budget = None
    if braw:
        try:
            budget = float(braw.replace("$", "").replace(",", ""))
            print(f"  ✓ Budget: ${budget:,.2f}")
        except ValueError:
            print("  Could not parse — no limit.")
 
    # 5. Build
    print("\n  Building deck …")
    deck = builder.build_deck(commander, seed_cards, budget,
                              external_seeds=ext_seeds,
                              archetype=archetype)
    builder.print_deck(deck)
 
    if input("\n  Save to CSV? (y/n): ").strip().lower() == "y":
        fname = input("  Filename [my_deck.csv]: ").strip() or "my_deck.csv"
        builder.export_deck(deck, fname)
 
 
# ─────────────────────────────────────────────
#  Demo mode
# ─────────────────────────────────────────────
def demo_build(builder):
    print("\n" + "=" * 60)
    print("  DEMO BUILD")
    print("=" * 60)
    cmdr = "Krenko Tin Street Kingpin"
    seeds = ["Sol Ring", "Lightning Bolt"]
    dids = builder.commander_deck_ids.get(cmdr, [])
    _, dom = builder.clusterer.commander_archetypes(dids)
    print(f"  Commander:  {cmdr}  [{builder._fmt_ci(cmdr)}]")
    print(f"  Seeds:      {seeds}")
    print(f"  Archetype:  #{dom+1}")
    print(f"  Budget:     No limit")
    deck = builder.build_deck(cmdr, seeds, budget=None, archetype=dom)
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