"""How well a brand's source library covers common campaign goals (keyword count, no model calls)."""
import re

GOALS = [
    ("Launch", ("launch", "launched", "launches", "introduce", "introduces", "introducing", "unveil", "unveils", "unveiled", "release", "released", "rollout", "debut")),
    ("Festive offer", ("festive", "festival", "diwali", "offer", "offers", "discount", "sale", "deal", "deals", "holiday", "christmas", "season", "seasonal")),
    ("Hiring", ("hiring", "career", "careers", "jobs", "recruit", "recruitment", "employee", "employees", "talent", "workplace", "culture", "join")),
    ("Sustainability", ("sustainability", "sustainable", "esg", "csr", "environment", "environmental", "carbon", "renewable", "green", "emissions", "responsibility")),
]
GOOD, ADEQUATE = 30, 10  # chunks


def status(n: int) -> str:
    return "Good" if n >= GOOD else ("Adequate" if n >= ADEQUATE else "Thin")


def coverage(chunk_texts: list[str]) -> list[dict]:
    sets = [set(re.findall(r"[a-z0-9]+", t.lower())) for t in chunk_texts]
    rows = []
    for name, terms in GOALS:
        n = sum(1 for s in sets if s.intersection(terms))
        rows.append({"Goal": name, "Chunks": n, "Status": status(n)})
    return rows
