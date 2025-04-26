from __future__ import annotations
import json
import itertools
import re
import textwrap
from pathlib import Path
from collections import Counter

import pandas as pd
import networkx as nx

DATA_FILE = Path("recipe.json")   
MIN_COOC_THRESHOLD = 2           
TOP_N_RELATED = 10               

STOPWORDS = {
    "fresh", "dried", "ground", "large", "small", "medium", "extra",
    "low-fat", "nonfat", "skinless", "boneless", "organic", "salted",
    "unsalted", "whole", "halved", "chopped", "finely", "minced",
    "crumbled", "shredded", "peeled", "ripe", "grated", "sliced",
    "frozen", "cold", "warm", "hot"
}

def normalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    tokens = [t for t in name.split() if t not in STOPWORDS]
    return " ".join(tokens)

def pairwise(iterable):
    return itertools.combinations(sorted(set(iterable)), 2)

class IngredientNetwork:
    def __init__(self, df: pd.DataFrame, min_cooc: int = MIN_COOC_THRESHOLD):
        self.df = df
        self.graph = nx.Graph()
        self._build_graph(min_cooc)

    def _build_graph(self, min_cooc: int):
        cooc = Counter()
        counts = Counter()

        for ingredients in self.df["ingredients"]:
            clean = [normalize(i) for i in ingredients]
            counts.update(clean)
            for a, b in pairwise(clean):
                cooc[(a, b)] += 1

        for ing, n in counts.items():
            self.graph.add_node(ing, count=n)

        for (a, b), w in cooc.items():
            if w >= min_cooc:
                self.graph.add_edge(a, b, weight=w)

    def related(self, ingredient: str, top_n: int = TOP_N_RELATED):
        ing = normalize(ingredient)
        if ing not in self.graph:
            return []
        nbrs = self.graph[ing]
        top = sorted(nbrs.items(), key=lambda it: it[1]["weight"], reverse=True)[:top_n]
        return [(n, d["weight"]) for n, d in top]

    def shortest_path(self, a: str, b: str):
        a, b = normalize(a), normalize(b)
        if a not in self.graph or b not in self.graph:
            return None
        try:
            return nx.shortest_path(self.graph, a, b)
        except nx.NetworkXNoPath:
            return None

    def most_connected(self):
        return max(self.graph.degree, key=lambda t: t[1]) 

    def stats(self, ingredient: str):
        ing = normalize(ingredient)
        if ing not in self.graph:
            return None
        return {
            "ingredient": ing,
            "recipes_containing": self.graph.nodes[ing]["count"],
            "degree": self.graph.degree[ing],
            "top_pairings": self.related(ing, 5)
        }

def load_dataset(path: Path | str = DATA_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{path}' not found. Download from Kaggle and place it here."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def build_network(data_path: Path | str = DATA_FILE,
                  min_cooc: int = MIN_COOC_THRESHOLD):
    df = load_dataset(data_path)
    return IngredientNetwork(df, min_cooc=min_cooc)

def _cli():
    net = build_network()
    print(f"Graph ready: {net.graph.number_of_nodes():,} ingredients, "
          f"{net.graph.number_of_edges():,} edges\n")

    MENU = textwrap.dedent("""\
        Choose an option:
          1) Related ingredients
          2) Shortest path between two ingredients
          3) Most connected ingredient
          4) Stats for an ingredient
          0) Quit
    """)
    while True:
        print(MENU)
        choice = input("Your choice: ").strip()
        if choice == "1":
            ing = input("Ingredient: ")
            for n, w in net.related(ing):
                print(f"  {n}  (co-occurs {w}×)")
            print()
        elif choice == "2":
            a = input("From ingredient: ")
            b = input("To ingredient: ")
            path = net.shortest_path(a, b)
            print("Path:" if path else "No connection.", path, "\n")
        elif choice == "3":
            ing, deg = net.most_connected()
            print(f"Most connected: {ing} (degree {deg})\n")
        elif choice == "4":
            ing = input("Ingredient: ")
            print(net.stats(ing) or "Not found", "\n")
        elif choice == "0":
            break
        else:
            print("Invalid choice.\n")

if __name__ == "__main__":
    _cli()
