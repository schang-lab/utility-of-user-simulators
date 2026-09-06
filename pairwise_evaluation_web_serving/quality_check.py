"""Heuristic quality check for pairwise study results.

Per-participant metrics:
  - query/word counts per conversation
  - on-topic score: keyword overlap between user queries and assigned intent
    (title + brief) plus prewriting answers.
  - AI-assistance heuristics: curly quotes, em-dashes, AI-jargon vocab,
    near-duplicate queries between conv 0 and conv 1 (suggests pasting from
    the same external draft), and per-turn word inflation.

Run:  python quality_check.py [results_dir]    (default: ./results)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

DEFAULT_DIR = Path(__file__).parent / "results"

STOPWORDS = set("""
a about above after again against all am an and any are aren as at be because been
before being below between both but by can could did do does doing don down during
each few for from further had has have having he her here hers herself him himself
his how i if in into is it its itself just me more most my myself no nor not now of
off on once only or other our ours ourselves out over own same she should so some
such than that the their theirs them themselves then there these they this those
through to too under until up very was we were what when where which while who whom
why will with would you your yours yourself yourselves
help write writing make get want need also like really one thing things little bit
more less please thanks thank ok okay yes
""".split())

AI_VOCAB = [
    # Stylometric tells common in LLM output
    "delve", "tapestry", "intricate", "intricately", "bustling", "pivotal",
    "navigating", "moreover", "furthermore", "in conclusion", "in essence",
    "it is important to note", "leverage", "leverages", "cultivate",
    "underscores", "underscore", "embark", "embarking", "myriad",
    "multifaceted", "robust", "seamless", "seamlessly", "elucidate",
    "harness", "endeavor", "esteemed", "noteworthy", "paramount",
    "synergy", "holistic", "ever-evolving", "in today's", "in the realm",
    "a testament to",
]

CURLY_QUOTES = "‘’“”"  # ' ' " "
EM_DASH = "—"  # —


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())


def content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def ai_marker_hits(text: str) -> dict[str, int]:
    low = text.lower()
    hits: dict[str, int] = {}
    for w in AI_VOCAB:
        c = low.count(w)
        if c:
            hits[w] = c
    n_curly = sum(text.count(c) for c in CURLY_QUOTES)
    if n_curly:
        hits["[curly_quotes]"] = n_curly
    n_em = text.count(EM_DASH)
    if n_em:
        hits["[em_dash]"] = n_em
    return hits


def collect_intent_keywords(p: dict) -> set[str]:
    setup = p.get("writing_setup") or {}
    intent = setup.get("intent") or {}
    parts = [intent.get("title", ""), intent.get("brief", "")]
    parts.extend(setup.get("prewriting_answers") or [])
    return content_tokens(" ".join(parts))


def all_queries(p: dict) -> list[tuple[int, int, str]]:
    """Return (conv_idx, turn_idx, query) tuples for every turn."""
    out = []
    for ci, conv in enumerate(p.get("conversations", [])):
        for ti, t in enumerate(conv.get("turns", [])):
            q = t.get("user_query") or ""
            out.append((ci, ti, q))
    return out


def near_duplicate_score(qs_a: list[str], qs_b: list[str]) -> float:
    """Max ratio of any conv-0 query against any conv-1 query (and vice versa)."""
    best = 0.0
    for a in qs_a:
        for b in qs_b:
            if not a or not b:
                continue
            r = SequenceMatcher(None, a, b, autojunk=False).ratio()
            if r > best:
                best = r
    return best


def analyze_participant(path: Path) -> dict:
    p = json.loads(path.read_text())
    pid = p.get("participant_id") or path.stem
    prolific_id = p.get("prolific_id")
    setup = p.get("writing_setup") or {}
    doc_type = setup.get("doc_type")
    intent = setup.get("intent") or {}
    intent_title = intent.get("title")
    intent_brief = intent.get("brief", "")
    intent_kw = collect_intent_keywords(p)

    queries = all_queries(p)
    n_queries = len(queries)
    word_counts = [len(q.split()) for _, _, q in queries]
    total_words = sum(word_counts)

    # On-topic = fraction of queries whose token set overlaps the intent keywords.
    on_topic_flags = []
    overlaps = []
    for _, _, q in queries:
        qtok = content_tokens(q)
        if not qtok:
            on_topic_flags.append(False)
            overlaps.append(0.0)
            continue
        ov = len(qtok & intent_kw) / max(1, len(qtok))
        overlaps.append(ov)
        on_topic_flags.append(ov >= 0.05 or len(qtok & intent_kw) >= 2)
    on_topic_rate = sum(on_topic_flags) / n_queries if n_queries else 0.0
    mean_overlap = sum(overlaps) / n_queries if n_queries else 0.0

    # AI markers, aggregated across queries.
    marker_totals: Counter[str] = Counter()
    per_query_marker_count = []
    for _, _, q in queries:
        h = ai_marker_hits(q)
        marker_totals.update(h)
        per_query_marker_count.append(sum(h.values()))
    n_queries_with_markers = sum(1 for c in per_query_marker_count if c)

    # Cross-conversation duplication.
    qs_by_conv: dict[int, list[str]] = {}
    for ci, _, q in queries:
        qs_by_conv.setdefault(ci, []).append(q)
    if len(qs_by_conv) >= 2:
        ks = sorted(qs_by_conv.keys())
        dup = near_duplicate_score(qs_by_conv[ks[0]], qs_by_conv[ks[1]])
    else:
        dup = 0.0

    # Heuristic flags.
    flags = []
    if doc_type is None or intent_title is None:
        flags.append("no_assigned_task")
    if n_queries == 0:
        flags.append("no_queries")
    if n_queries and on_topic_rate < 0.5:
        flags.append(f"off_topic({on_topic_rate:.0%})")
    if marker_totals.get("[curly_quotes]", 0) >= 4:
        flags.append("curly_quotes")
    if marker_totals.get("[em_dash]", 0) >= 2:
        flags.append("em_dash")
    n_jargon = sum(v for k, v in marker_totals.items() if not k.startswith("["))
    if n_jargon >= 2:
        flags.append(f"ai_vocab({n_jargon})")
    if dup >= 0.85:
        flags.append(f"near_dup_convs({dup:.2f})")
    if word_counts and (sum(word_counts) / len(word_counts)) >= 80:
        flags.append("long_queries")
    if not p.get("completed") or "--incomplete" in path.name:
        flags.append("incomplete")

    return {
        "file": path.name,
        "pid": pid,
        "prolific_id": prolific_id,
        "doc_type": doc_type,
        "intent_title": intent_title,
        "intent_brief": intent_brief,
        "completed": bool(p.get("completed")),
        "n_conversations": len(p.get("conversations", [])),
        "n_queries": n_queries,
        "total_query_words": total_words,
        "mean_words_per_query": (total_words / n_queries) if n_queries else 0.0,
        "median_words_per_query": (sorted(word_counts)[len(word_counts)//2] if word_counts else 0),
        "max_words_per_query": max(word_counts) if word_counts else 0,
        "min_words_per_query": min(word_counts) if word_counts else 0,
        "on_topic_rate": on_topic_rate,
        "mean_overlap": mean_overlap,
        "ai_marker_totals": dict(marker_totals),
        "n_queries_with_markers": n_queries_with_markers,
        "cross_conv_max_similarity": dup,
        "flags": flags,
        "queries": [
            {"conv": ci, "turn": ti, "words": len(q.split()), "query": q}
            for (ci, ti, q) in queries
        ],
    }


def main(results_dir: Path) -> None:
    files = sorted(results_dir.glob("*.json"))
    files = [f for f in files if not f.name.startswith("_")]
    if not files:
        print(f"no participant JSONs in {results_dir}", file=sys.stderr)
        sys.exit(1)

    rows = [analyze_participant(f) for f in files]

    print(f"\n=== Quality check: {results_dir} ===")
    print(f"participants: {len(rows)}  "
          f"completed: {sum(r['completed'] for r in rows)}  "
          f"incomplete: {sum(1 for r in rows if 'incomplete' in r['flags'])}")

    total_queries = sum(r["n_queries"] for r in rows)
    total_words = sum(r["total_query_words"] for r in rows)
    if total_queries:
        print(f"total queries: {total_queries}  "
              f"total query words: {total_words}  "
              f"avg words/query: {total_words/total_queries:.1f}")

    # Doc-type / intent distribution
    print("\n--- Assigned intents ---")
    intents = Counter(((r["doc_type"], r["intent_title"]) for r in rows))
    for (dt, title), n in intents.most_common():
        print(f"  {n:>2}  {dt:<22} {title}")

    # Words-per-query summary
    print("\n--- Per-participant summary (sorted by on_topic_rate asc) ---")
    fmt = "{pid:<10} {n_q:>3}q {mean_w:>5.1f}w  on_topic={ot:>5.0%}  "\
          "dup={dup:>4.2f}  markers={mk:<3} flags={flags}"
    for r in sorted(rows, key=lambda x: (x["on_topic_rate"], -x["n_queries"])):
        print(fmt.format(
            pid=r["pid"][:8],
            n_q=r["n_queries"],
            mean_w=r["mean_words_per_query"],
            ot=r["on_topic_rate"],
            dup=r["cross_conv_max_similarity"],
            mk=r["n_queries_with_markers"],
            flags=",".join(r["flags"]) if r["flags"] else "-",
        ))

    # Flagged participants
    flagged = [r for r in rows if r["flags"] and r["flags"] != ["incomplete"]]
    print(f"\n--- Flagged participants ({len(flagged)}) ---")
    for r in sorted(flagged, key=lambda x: -len(x["flags"])):
        print(f"\n  pid={r['pid']}  prolific={r['prolific_id']}")
        print(f"    intent  = {r['doc_type']} / {r['intent_title']!r}")
        print(f"    flags   = {', '.join(r['flags'])}")
        print(f"    n_queries={r['n_queries']}  mean_words={r['mean_words_per_query']:.1f}  "
              f"on_topic_rate={r['on_topic_rate']:.0%}  dup={r['cross_conv_max_similarity']:.2f}")
        if r["ai_marker_totals"]:
            print(f"    markers = {r['ai_marker_totals']}")
        # Show every query so a human reviewer can confirm.
        for q in r["queries"]:
            preview = q["query"].replace("\n", " ")
            if len(preview) > 220:
                preview = preview[:220] + "..."
            print(f"      c{q['conv']}t{q['turn']} [{q['words']:>3}w] {preview}")

    # Write JSON dump for downstream use.
    out_path = results_dir / "_quality_check.json"
    out_path.write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "queries"} for r in rows],
        indent=2,
    ))
    print(f"\nwrote summary -> {out_path}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    main(target)
