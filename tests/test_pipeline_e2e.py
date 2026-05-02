import json
from pathlib import Path

from eval_script import evaluate_results, normalize_std
from run import run_file
from src.pipeline import BISPipeline
from src.retriever import normalize_standard_code


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_public_set_metrics(tmp_path):
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

    output = tmp_path / "results.json"
    output.write_text(json.dumps(results), encoding="utf-8")
    evaluate_results(output)

    hits_at_3 = 0
    reciprocal_rank_sum = 0.0
    for item in results:
        expected = {normalize_std(code) for code in item["expected_standards"]}
        retrieved = [normalize_std(code) for code in item["retrieved_standards"]]
        if any(code in expected for code in retrieved[:3]):
            hits_at_3 += 1
        rank = next((index for index, code in enumerate(retrieved[:5], start=1) if code in expected), None)
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank

    assert hits_at_3 / len(results) >= 0.9
    assert reciprocal_rank_sum / len(results) >= 0.85


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
