#!/usr/bin/env python3
"""BlackRoad AI Classifier — keyword-weighted text classification with
confidence scoring and SQLite persistence."""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

GREEN  = "\033[0;32m"
RED    = "\033[0;31m"
YELLOW = "\033[1;33m"
CYAN   = "\033[0;36m"
BOLD   = "\033[1m"
NC     = "\033[0m"

DB_PATH = Path.home() / ".blackroad" / "ai_classifier.db"


# ── Data models ───────────────────────────────────────────────────────────────
@dataclass
class Category:
    id: Optional[int]
    name: str
    description: str
    keywords: str           # JSON: list of [keyword, weight] pairs
    colour: str
    examples_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class TrainingExample:
    id: Optional[int]
    category_name: str
    text: str
    source: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ClassificationResult:
    id: Optional[int]
    input_text: str
    predicted_category: str
    confidence: float           # 0.0 – 1.0
    runner_up: str
    runner_up_confidence: float
    all_scores: str             # JSON dict of cat -> probability
    model_version: str
    classified_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def confidence_bar(self, width: int = 20) -> str:
        filled = int(self.confidence * width)
        bar = "\u2588" * filled + "\u2591" * (width - filled)
        colour = (GREEN if self.confidence >= 0.7
                  else YELLOW if self.confidence >= 0.4
                  else RED)
        return f"{colour}{bar}{NC} {self.confidence:.1%}"


# ── Database ──────────────────────────────────────────────────────────────────
def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS categories ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  name TEXT NOT NULL UNIQUE,"
        "  description TEXT DEFAULT '',"
        "  keywords TEXT DEFAULT '[]',"
        "  colour TEXT DEFAULT '\033[0;36m',"
        "  examples_count INTEGER DEFAULT 0,"
        "  created_at TEXT NOT NULL"
        ");"
        "CREATE TABLE IF NOT EXISTS training_examples ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  category_name TEXT NOT NULL,"
        "  text TEXT NOT NULL,"
        "  source TEXT DEFAULT '',"
        "  created_at TEXT NOT NULL"
        ");"
        "CREATE TABLE IF NOT EXISTS classification_results ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  input_text TEXT NOT NULL,"
        "  predicted_category TEXT NOT NULL,"
        "  confidence REAL NOT NULL,"
        "  runner_up TEXT DEFAULT '',"
        "  runner_up_confidence REAL DEFAULT 0,"
        "  all_scores TEXT DEFAULT '{}',"
        "  model_version TEXT DEFAULT 'v1',"
        "  classified_at TEXT NOT NULL"
        ");"
        "CREATE INDEX IF NOT EXISTS idx_results_cat"
        "  ON classification_results(predicted_category);"
    )
    conn.commit()
    return conn


# ── Scoring utilities ─────────────────────────────────────────────────────────
def _tokenise(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _score_category(tokens: List[str],
                    keywords: List[Tuple[str, float]]) -> float:
    """TF-weighted keyword scoring against a category's keyword list."""
    if not keywords:
        return 0.0
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    score = 0.0
    for kw, weight in keywords:
        kw_lower = kw.lower()
        if kw_lower in freq:
            score += weight * math.log1p(freq[kw_lower])
    return score


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    max_v = max(scores.values())
    exp_s = {k: math.exp(v - max_v) for k, v in scores.items()}
    total = sum(exp_s.values())
    return {k: v / total for k, v in exp_s.items()}


# ── Classifier ────────────────────────────────────────────────────────────────
class AIClassifier:
    """Keyword-weighted multi-class text classifier with SQLite persistence."""

    MODEL_VERSION = "keyword-weighted-v1.2"

    def __init__(self) -> None:
        self._db = _get_db()
        self._cache: Optional[List[Tuple[str, List[Tuple[str, float]]]]] = None

    def _load_categories(self) -> List[Tuple[str, List[Tuple[str, float]]]]:
        if self._cache is None:
            rows = self._db.execute(
                "SELECT name, keywords FROM categories").fetchall()
            self._cache = [
                (r["name"], json.loads(r["keywords"])) for r in rows
            ]
        return self._cache

    def _invalidate(self) -> None:
        self._cache = None

    # ── Category management ────────────────────────────────────────────────
    def add_category(
            self,
            name: str,
            description: str = "",
            keywords: Optional[List[Tuple[str, float]]] = None,
            colour: str = CYAN) -> Category:
        """Upsert a classification category with weighted keywords."""
        kw_json = json.dumps(keywords or [])
        ts = datetime.utcnow().isoformat()
        existing = self._db.execute(
            "SELECT id FROM categories WHERE name=?", (name,)
        ).fetchone()
        if existing:
            self._db.execute(
                "UPDATE categories"
                " SET description=?, keywords=?, colour=? WHERE name=?",
                (description, kw_json, colour, name),
            )
            self._db.commit()
            cid = existing["id"]
        else:
            cur = self._db.execute(
                "INSERT INTO categories"
                " (name, description, keywords, colour, created_at)"
                " VALUES (?,?,?,?,?)",
                (name, description, kw_json, colour, ts),
            )
            self._db.commit()
            cid = cur.lastrowid
        self._invalidate()
        return Category(id=cid, name=name, description=description,
                        keywords=kw_json, colour=colour, created_at=ts)

    def add_example(self, category: str, text: str,
                    source: str = "") -> TrainingExample:
        """Store a labelled training example and bump the category counter."""
        ts = datetime.utcnow().isoformat()
        cur = self._db.execute(
            "INSERT INTO training_examples"
            " (category_name, text, source, created_at) VALUES (?,?,?,?)",
            (category, text, source, ts),
        )
        self._db.execute(
            "UPDATE categories SET examples_count=examples_count+1"
            " WHERE name=?",
            (category,),
        )
        self._db.commit()
        self._invalidate()
        return TrainingExample(id=cur.lastrowid, category_name=category,
                               text=text, source=source, created_at=ts)

    # ── Classification ─────────────────────────────────────────────────────
    def classify(self, text: str) -> ClassificationResult:
        """Run multi-class classification and return a scored result."""
        categories = self._load_categories()
        if not categories:
            raise RuntimeError(
                "No categories defined. Use 'add category' first.")
        tokens = _tokenise(text)
        raw = {name: _score_category(tokens, kws) + 1e-6
               for name, kws in categories}
        probs = _softmax(raw)
        ranked = sorted(probs.items(), key=lambda x: -x[1])
        best_cat, best_conf  = ranked[0]
        runner, runner_conf  = ranked[1] if len(ranked) > 1 else ("", 0.0)

        ts = datetime.utcnow().isoformat()
        top_scores = {k: round(v, 4) for k, v in ranked}
        cur = self._db.execute(
            "INSERT INTO classification_results"
            " (input_text, predicted_category, confidence,"
            "  runner_up, runner_up_confidence, all_scores,"
            "  model_version, classified_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (text[:500], best_cat, round(best_conf, 4),
             runner, round(runner_conf, 4),
             json.dumps(top_scores),
             self.MODEL_VERSION, ts),
        )
        self._db.commit()
        return ClassificationResult(
            id=cur.lastrowid, input_text=text,
            predicted_category=best_cat, confidence=round(best_conf, 4),
            runner_up=runner, runner_up_confidence=round(runner_conf, 4),
            all_scores=json.dumps(top_scores),
            model_version=self.MODEL_VERSION, classified_at=ts,
        )

    # ── Listing & stats ────────────────────────────────────────────────────
    def list_results(self, limit: int = 25) -> List[dict]:
        rows = self._db.execute(
            "SELECT id, predicted_category, confidence,"
            "       input_text, classified_at"
            " FROM classification_results"
            " ORDER BY classified_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_categories(self) -> List[dict]:
        rows = self._db.execute(
            "SELECT name, description, examples_count, created_at, keywords"
            " FROM categories ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]

    def pipeline_stats(self) -> dict:
        db = self._db
        total    = db.execute(
            "SELECT COUNT(*) FROM classification_results").fetchone()[0]
        cats     = db.execute(
            "SELECT COUNT(*) FROM categories").fetchone()[0]
        examples = db.execute(
            "SELECT COUNT(*) FROM training_examples").fetchone()[0]
        avg_row  = db.execute(
            "SELECT AVG(confidence) FROM classification_results").fetchone()[0]
        return {
            "total_classified":  total,
            "categories":        cats,
            "training_examples": examples,
            "avg_confidence":    round(avg_row or 0.0, 4),
        }

    def export(self, output: str = "classifier_export.json") -> Path:
        """Dump categories and classification history to JSON."""
        results = self._db.execute(
            "SELECT * FROM classification_results"
            " ORDER BY classified_at").fetchall()
        cats = self._db.execute(
            "SELECT * FROM categories").fetchall()
        data = {
            "exported_at":   datetime.utcnow().isoformat(),
            "model_version": self.MODEL_VERSION,
            "categories":    [dict(r) for r in cats],
            "results":       [dict(r) for r in results],
        }
        out = Path(output)
        out.write_text(json.dumps(data, indent=2))
        return out


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ai_classifier",
        description=(f"{BOLD}BlackRoad AI Classifier{NC}"
                     " — text classification with confidence scoring"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", metavar="command")

    p_list = sub.add_parser("list", help="List results or categories")
    p_list.add_argument(
        "--type", choices=["results", "categories"], default="results")
    p_list.add_argument("--limit", type=int, default=25)

    p_add = sub.add_parser("add", help="Add a category or training example")
    p_add.add_argument("type", choices=["category", "example"])
    p_add.add_argument("name",
                       help="Category name, or text body for an example")
    p_add.add_argument("--description", default="")
    p_add.add_argument("--keywords",    default="[]",
                       help='JSON e.g. [["word", 2.5], ["term", 1.0]]')
    p_add.add_argument("--category",    default="",
                       help="Target category (required for type=example)")
    p_add.add_argument("--source",      default="")

    sub.add_parser("status", help="Show classifier pipeline stats")

    p_classify = sub.add_parser("classify", help="Classify input text")
    p_classify.add_argument("text", help="Text to classify")

    p_export = sub.add_parser("export", help="Export all data to JSON")
    p_export.add_argument("--output", default="classifier_export.json")

    args = parser.parse_args()
    clf = AIClassifier()

    if args.cmd == "list":
        if args.type == "categories":
            cats = clf.list_categories()
            print(f"\n{BOLD}{CYAN}\U0001f3f7\ufe0f  Categories ({len(cats)}){NC}\n")
            for c in cats:
                kws = json.loads(c["keywords"])
                kw_preview = ", ".join(k for k, _ in kws[:5])
                print(f"  {BOLD}{c['name']:<24}{NC}"
                      f"  examples={c['examples_count']:<5}"
                      f"  keywords: {kw_preview or '(none)'}")
        else:
            rows = clf.list_results(args.limit)
            print(f"\n{BOLD}{CYAN}\U0001f916 Classification Results"
                  f" ({len(rows)}){NC}\n")
            for r in rows:
                ts = r["classified_at"][:19].replace("T", " ")
                conf = r["confidence"]
                colour = (GREEN if conf >= 0.7
                          else YELLOW if conf >= 0.4 else RED)
                preview = r["input_text"][:48].replace("\n", " ")
                print(f"  {colour}{r['predicted_category']:<22}{NC}"
                      f"  {conf:.1%}  {preview!r}  {ts}")
        print()

    elif args.cmd == "add":
        if args.type == "category":
            try:
                kws = json.loads(args.keywords)
            except json.JSONDecodeError:
                print(f"{RED}\u2717 --keywords must be valid JSON{NC}",
                      file=sys.stderr)
                sys.exit(1)
            cat = clf.add_category(
                args.name,
                description=args.description,
                keywords=kws,
            )
            print(f"{GREEN}\u2705 Category '{cat.name}'{NC}"
                  f"  {len(kws)} keywords  desc: {cat.description or '—'}")
        else:
            if not args.category:
                print(
                    f"{RED}\u2717 --category required for type=example{NC}",
                    file=sys.stderr,
                )
                sys.exit(1)
            ex = clf.add_example(
                args.category, args.name, source=args.source)
            print(f"{GREEN}\u2705 Example added{NC}"
                  f"  id={ex.id}  \u2192 {ex.category_name}")

    elif args.cmd == "status":
        s = clf.pipeline_stats()
        print(f"\n{BOLD}{CYAN}\U0001f4ca Classifier — Pipeline Stats{NC}\n")
        print(f"  Categories        : {BOLD}{s['categories']}{NC}")
        print(f"  Training examples : {s['training_examples']}")
        print(f"  Total classified  : {s['total_classified']}")
        print(f"  Avg confidence    :"
              f" {BOLD}{s['avg_confidence']:.1%}{NC}\n")

    elif args.cmd == "classify":
        try:
            result = clf.classify(args.text)
        except RuntimeError as exc:
            print(f"{RED}\u2717 {exc}{NC}", file=sys.stderr)
            sys.exit(1)
        print(f"\n{BOLD}{CYAN}\U0001f916 Classification Result{NC}\n")
        print(f"  Input     : {args.text[:80]!r}")
        print(f"  Category  : {BOLD}{result.predicted_category}{NC}")
        print(f"  Confidence: {result.confidence_bar()}")
        if result.runner_up:
            print(f"  Runner-up : {result.runner_up}"
                  f"  ({result.runner_up_confidence:.1%})")
        scores = json.loads(result.all_scores)
        if len(scores) > 1:
            print(f"\n  {BOLD}All scores:{NC}")
            for cat, score in list(scores.items())[:8]:
                filled = int(score * 20)
                bar = "\u2588" * filled + "\u2591" * (20 - filled)
                print(f"  {cat:<24}  {CYAN}{bar}{NC}  {score:.1%}")
        print()

    elif args.cmd == "export":
        out = clf.export(args.output)
        print(f"{GREEN}\u2705 Exported \u2192{NC} {out}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
