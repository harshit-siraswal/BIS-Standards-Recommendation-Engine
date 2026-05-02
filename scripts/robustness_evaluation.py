"""Run a broad robustness evaluation for the BIS recommendation pipeline.

The generated dataset is synthetic but label-controlled: every query is a
paraphrase, translation, typo/noise variant, or underspecified variant of one of
the public-test target standards.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import BISPipeline
from src.retriever import normalize_standard_code


STANDARDS: list[dict[str, str]] = [
    {
        "code": "IS 269: 1989",
        "name": "33 Grade Ordinary Portland Cement",
        "keywords": "33 grade ordinary portland cement OPC chemical physical requirements",
        "short": "OPC 33 grade",
        "hinglish": "33 grade OPC cement ka BIS standard kya hai",
        "hindi": "33 ग्रेड साधारण पोर्टलैंड सीमेंट के लिए कौन सा BIS मानक है",
        "spanish": "cemento Portland ordinario grado 33 requisitos quimicos fisicos",
        "french": "ciment Portland ordinaire grade 33 exigences chimiques physiques",
        "tamil": "33 grade ordinary Portland cement standard enna",
        "vague": "normal cement grade 33",
    },
    {
        "code": "IS 383: 1970",
        "name": "Coarse and Fine Aggregates",
        "keywords": "coarse fine aggregates natural sources structural concrete",
        "short": "natural aggregate concrete",
        "hinglish": "concrete ke liye coarse fine aggregate standard",
        "hindi": "संरचनात्मक कंक्रीट के लिए प्राकृतिक मोटा और महीन एग्रीगेट मानक",
        "spanish": "agregados gruesos y finos de fuentes naturales para concreto estructural",
        "french": "granulats grossiers et fins naturels pour beton structurel",
        "tamil": "structural concrete coarse fine aggregate standard enna",
        "vague": "sand gravel for concrete",
    },
    {
        "code": "IS 458: 2003",
        "name": "Precast Concrete Pipes",
        "keywords": "precast concrete pipes reinforced unreinforced water mains",
        "short": "RCC concrete pipe water main",
        "hinglish": "paani main ke liye precast concrete pipe BIS code",
        "hindi": "जल मुख्य लाइन के लिए प्रीकास्ट कंक्रीट पाइप का मानक",
        "spanish": "tubos de concreto prefabricado para conducciones de agua con refuerzo",
        "french": "tuyaux en beton prefabrique pour conduites d eau avec armature",
        "tamil": "precast concrete pipe water main standard enna",
        "vague": "cement pipe for water line",
    },
    {
        "code": "IS 2185 (Part 2): 1983",
        "name": "Lightweight Concrete Masonry Blocks",
        "keywords": "hollow solid lightweight concrete masonry blocks dimensions physical requirements",
        "short": "lightweight hollow solid blocks",
        "hinglish": "hollow solid lightweight concrete block ka standard",
        "hindi": "हल्के खोखले और ठोस कंक्रीट masonry block का मानक",
        "spanish": "bloques huecos y solidos de concreto ligero requisitos dimensionales",
        "french": "blocs de beton leger creux et pleins dimensions exigences",
        "tamil": "lightweight hollow solid concrete block standard enna",
        "vague": "light blocks for masonry wall",
    },
    {
        "code": "IS 459: 1992",
        "name": "Corrugated Asbestos Cement Sheets",
        "keywords": "corrugated semi corrugated asbestos cement sheets roofing cladding",
        "short": "AC sheet roofing",
        "hinglish": "roofing cladding ke liye asbestos cement sheet standard",
        "hindi": "छत और cladding के लिए corrugated asbestos cement sheet मानक",
        "spanish": "laminas onduladas de asbesto cemento para techo y revestimiento",
        "french": "plaques ondulees en amiante ciment pour toiture bardage",
        "tamil": "asbestos cement roofing sheet standard enna",
        "vague": "corrugated roof sheet",
    },
    {
        "code": "IS 455: 1989",
        "name": "Portland Slag Cement",
        "keywords": "portland slag cement manufacture chemical physical requirements",
        "short": "PSC slag cement",
        "hinglish": "portland slag cement ka BIS code kya hai",
        "hindi": "पोर्टलैंड स्लैग सीमेंट के निर्माण और आवश्यकताओं का मानक",
        "spanish": "cemento Portland con escoria requisitos de fabricacion",
        "french": "ciment Portland au laitier exigences fabrication",
        "tamil": "Portland slag cement standard enna",
        "vague": "slag based cement",
    },
    {
        "code": "IS 1489 (Part 2): 1991",
        "name": "Calcined Clay Portland Pozzolana Cement",
        "keywords": "Portland pozzolana cement calcined clay based PPC part 2",
        "short": "PPC calcined clay",
        "hinglish": "calcined clay based PPC ke liye kaunsa IS code",
        "hindi": "calcined clay आधारित पोर्टलैंड pozzolana cement का मानक",
        "spanish": "cemento Portland puzolana basado en arcilla calcinada",
        "french": "ciment Portland pouzzolane a base d argile calcinee",
        "tamil": "calcined clay PPC standard enna",
        "vague": "pozzolana cement clay based",
    },
    {
        "code": "IS 3466: 1988",
        "name": "Masonry Cement",
        "keywords": "masonry cement mortar general purpose non structural concrete",
        "short": "masonry cement mortar",
        "hinglish": "masonry mortar ke liye cement standard kya hai",
        "hindi": "masonry mortar के लिए सामान्य उपयोग वाला cement मानक",
        "spanish": "cemento de albanileria para morteros no concreto estructural",
        "french": "ciment de maconnerie pour mortiers usage general",
        "tamil": "masonry cement mortar standard enna",
        "vague": "cement for brick mortar",
    },
    {
        "code": "IS 6909: 1990",
        "name": "Supersulphated Cement",
        "keywords": "supersulphated cement marine works aggressive water conditions",
        "short": "SSC marine cement",
        "hinglish": "marine works aggressive water ke liye supersulphated cement standard",
        "hindi": "समुद्री कार्य और aggressive water के लिए supersulphated cement मानक",
        "spanish": "cemento supersulfatado para obras marinas aguas agresivas",
        "french": "ciment sursulfate pour travaux marins eaux agressives",
        "tamil": "supersulphated cement marine works standard enna",
        "vague": "cement for sea water exposure",
    },
    {
        "code": "IS 8042: 1989",
        "name": "White Portland Cement",
        "keywords": "white Portland cement architectural decorative chemical physical requirements",
        "short": "white cement decorative",
        "hinglish": "white portland cement decoration ke liye standard",
        "hindi": "सजावटी काम के लिए सफेद पोर्टलैंड सीमेंट मानक",
        "spanish": "cemento Portland blanco para usos arquitectonicos decorativos",
        "french": "ciment Portland blanc pour usages architecturaux decoratifs",
        "tamil": "white Portland cement decorative standard enna",
        "vague": "white decorative cement",
    },
]


def build_queries() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, standard in enumerate(STANDARDS, start=1):
        templates = [
            ("english_public_like", f"We manufacture {standard['name']}. Which BIS standard covers it?"),
            ("english_public_like", f"What Indian Standard applies to {standard['keywords']}?"),
            ("english_public_like", f"Need specification for {standard['name']} including requirements and testing."),
            ("english_public_like", f"Find the IS code for a product described as {standard['keywords']}."),
            ("short_telegraphic", standard["short"]),
            ("short_telegraphic", f"{standard['short']} BIS"),
            ("short_telegraphic", f"{standard['short']} standard"),
            ("underspecified", standard["vague"]),
            ("underspecified", f"which standard for {standard['vague']}"),
            ("noisy_typos", typo_variant(standard["keywords"])),
            ("noisy_typos", typo_variant(standard["short"] + " specification")),
            ("direct_code", f"Tell me about {standard['code']}"),
            ("hinglish", standard["hinglish"]),
            ("hinglish", f"{standard['short']} ka IS code batao"),
            ("hindi_devanagari", standard["hindi"]),
            ("hindi_devanagari", f"{standard['hindi']} कौन सा है"),
            ("spanish", standard["spanish"]),
            ("spanish", f"norma BIS para {standard['spanish']}"),
            ("french", standard["french"]),
            ("tamil_transliterated", standard["tamil"]),
        ]
        for variant_index, (style, query) in enumerate(templates, start=1):
            rows.append(
                {
                    "id": f"ROB-{index:02d}-{variant_index:02d}",
                    "query": query,
                    "expected_standards": [standard["code"]],
                    "target_name": standard["name"],
                    "style": style,
                }
            )
    return rows


def typo_variant(text: str) -> str:
    replacements = {
        "requirements": "reqrmnts",
        "physical": "phisical",
        "chemical": "chemicl",
        "cement": "cemnt",
        "concrete": "concret",
        "standard": "std",
        "portland": "portlnd",
        "aggregate": "agregate",
        "aggregates": "agregates",
        "masonry": "masonary",
        "corrugated": "corugated",
        "reinforced": "reinfrced",
    }
    noisy = text.lower()
    for src, dst in replacements.items():
        noisy = noisy.replace(src, dst)
    return noisy


def reciprocal_rank(retrieved: list[str], expected: set[str], limit: int = 5) -> float:
    for rank, code in enumerate(retrieved[:limit], start=1):
        if normalize_standard_code(code) in expected:
            return 1.0 / rank
    return 0.0


def run_evaluation(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = build_queries()
    pipeline = BISPipeline(use_reranker=False)

    results: list[dict[str, Any]] = []
    for row in queries:
        out = pipeline.run_query(row["query"])
        expected_norm = {normalize_standard_code(code) for code in row["expected_standards"]}
        retrieved = out["retrieved_standards"]
        retrieved_norm = [normalize_standard_code(code) for code in retrieved]
        rr = reciprocal_rank(retrieved, expected_norm)
        rank = 0
        for i, code in enumerate(retrieved_norm, start=1):
            if code in expected_norm:
                rank = i
                break
        results.append(
            {
                **row,
                "retrieved_standards": retrieved,
                "latency_seconds": out["latency_seconds"],
                "rank": rank,
                "hit_at_1": rank == 1,
                "hit_at_3": 1 <= rank <= 3,
                "hit_at_5": 1 <= rank <= 5,
                "reciprocal_rank": rr,
                "out_of_scope": out.get("out_of_scope", False),
            }
        )

    summary = summarize(results)
    (output_dir / "robustness_queries.json").write_text(
        json.dumps(queries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "robustness_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "robustness_results.csv", results)
    return summary


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    latencies = [float(row["latency_seconds"]) for row in results]
    summary: dict[str, Any] = {
        "total_queries": total,
        "hit_rate_at_1": round(sum(row["hit_at_1"] for row in results) / total * 100, 2),
        "hit_rate_at_3": round(sum(row["hit_at_3"] for row in results) / total * 100, 2),
        "hit_rate_at_5": round(sum(row["hit_at_5"] for row in results) / total * 100, 2),
        "mrr_at_5": round(sum(row["reciprocal_rank"] for row in results) / total, 4),
        "avg_latency_seconds": round(statistics.mean(latencies), 4),
        "p95_latency_seconds": round(percentile(latencies, 95), 4),
        "max_latency_seconds": round(max(latencies), 4),
        "by_style": {},
        "by_target": {},
        "rank_distribution": dict(sorted(Counter(row["rank"] for row in results).items())),
    }

    for key, out_key in (("style", "by_style"), ("target_name", "by_target")):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in results:
            groups[str(row[key])].append(row)
        for group, rows in sorted(groups.items()):
            count = len(rows)
            summary[out_key][group] = {
                "queries": count,
                "hit_rate_at_3": round(sum(row["hit_at_3"] for row in rows) / count * 100, 2),
                "mrr_at_5": round(sum(row["reciprocal_rank"] for row in rows) / count, 4),
                "top1": round(sum(row["hit_at_1"] for row in rows) / count * 100, 2),
            }
    return summary


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "style",
        "target_name",
        "query",
        "expected_standards",
        "retrieved_standards",
        "rank",
        "hit_at_3",
        "reciprocal_rank",
        "latency_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/evaluation", help="Directory for evaluation artifacts.")
    args = parser.parse_args()

    summary = run_evaluation(Path(args.output_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
