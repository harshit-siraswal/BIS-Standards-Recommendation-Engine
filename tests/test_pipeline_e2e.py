import json
from pathlib import Path

from eval_script import evaluate_results
from run import run_file
from src.pipeline import BISPipeline
from src.retriever import normalize_standard_code


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_public_set_metrics():
    test_set = json.loads((ROOT / "public_test_set.json").read_text(encoding="utf-8"))
    pipeline = BISPipeline()
    results = []
    for item in test_set:
        out = pipeline.run_query(item["query"])
        results.append(
            {
                **item,
                "retrieved_standards": out["retrieved_standards"],
                "latency_seconds": out["latency_seconds"],
            }
        )

    metrics = evaluate_results(results)
    assert metrics["hit_rate_at_3"] >= 0.9
    assert metrics["mrr_at_5"] >= 0.85


def test_run_file_writes_expected_schema(tmp_path):
    output = tmp_path / "results.json"
    results = run_file(ROOT / "public_test_set.json", output)

    assert output.exists()
    assert len(results) == 10
    assert all(len(row["retrieved_standards"]) == 5 for row in results)


def test_pipeline_handles_empty_query_with_five_defaults():
    out = BISPipeline().run_query("")

    assert len(out["retrieved_standards"]) == 5
    assert normalize_standard_code(out["retrieved_standards"][0]) == "is269:1989"
