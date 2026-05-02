import re

from fastapi.testclient import TestClient

from app import INDEX_HTML, _bis_portal_search_url_for_code, app


client = TestClient(app)


def test_status_endpoint_renders_page_for_browser():
    response = client.get("/api/status", headers={"accept": "text/html"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Service Status" in response.text
    assert "Search standards" in response.text


def test_status_endpoint_keeps_json_mode_for_monitors():
    response = client.get("/api/status?format=json", headers={"accept": "text/html"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["status"] == "ready"


def test_favicon_svg_is_served():
    response = client.get("/favicon.svg")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert "BIS" in response.text


def test_language_map_covers_all_interface_text_keys():
    base_match = re.search(r"const baseText = \{(.*?)\n    \};", INDEX_HTML, re.S)
    assert base_match is not None

    base_keys = set(re.findall(r"\n\s{6}([A-Za-z0-9_]+):", base_match.group(1)))
    assert base_keys

    for language in ["hi", "hinglish", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur"]:
        language_keys = set()
        for block in re.finditer(rf"\n\s{{6}}{language}: \{{(.*?)\n\s{{6}}\}}", INDEX_HTML, re.S):
            language_keys.update(re.findall(r"\n\s{8}([A-Za-z0-9_]+):", block.group(1)))

        assert base_keys - language_keys == set(), language


def test_multilingual_pencil_queries_return_verified_external_standards():
    queries = [
        ("en", "we are making graphite lead pencils"),
        ("hi", "\u0939\u092e \u0917\u094d\u0930\u0947\u092b\u093e\u0907\u091f \u0932\u0940\u0921 \u092a\u0947\u0902\u0938\u093f\u0932 \u092c\u0928\u093e\u0924\u0947 \u0939\u0948\u0902"),
        ("hinglish", "hum graphite lead pencils bana rahe hain"),
        ("bn", "\u0986\u09ae\u09b0\u09be \u0997\u09cd\u09b0\u09be\u09ab\u09be\u0987\u099f \u09b2\u09c7\u09a1 \u09aa\u09c7\u09a8\u09cd\u09b8\u09bf\u09b2 \u09a4\u09c8\u09b0\u09bf \u0995\u09b0\u09bf"),
        ("ta", "\u0ba8\u0bbe\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0b95\u0bbf\u0bb0\u0bbe\u0b83\u0baa\u0bc8\u0b9f\u0bcd \u0bb2\u0bc0\u0b9f\u0bcd \u0baa\u0bc6\u0ba9\u0bcd\u0b9a\u0bbf\u0bb2\u0bcd\u0b95\u0bb3\u0bcd \u0ba4\u0baf\u0bbe\u0bb0\u0bbf\u0b95\u0bcd\u0b95\u0bbf\u0bb1\u0bcb\u0bae\u0bcd"),
        ("te", "\u0c2e\u0c47\u0c2e\u0c41 \u0c17\u0c4d\u0c30\u0c3e\u0c2b\u0c48\u0c1f\u0c4d \u0c32\u0c40\u0c21\u0c4d \u0c2a\u0c46\u0c28\u0c4d\u0c38\u0c3f\u0c32\u0c4d\u0c38\u0c4d \u0c24\u0c2f\u0c3e\u0c30\u0c41 \u0c1a\u0c47\u0c38\u0c4d\u0c24\u0c41\u0c28\u0c4d\u0c28\u0c3e\u0c2e\u0c41"),
        ("mr", "\u0906\u092e\u094d\u0939\u0940 \u0917\u094d\u0930\u0947\u092b\u093e\u0907\u091f \u0932\u0940\u0921 \u092a\u0947\u0928\u094d\u0938\u093f\u0932 \u092c\u0928\u0935\u0924\u094b"),
        ("gu", "\u0a85\u0aae\u0ac7 \u0a97\u0acd\u0ab0\u0abe\u0aab\u0abe\u0a87\u0a9f \u0ab2\u0ac0\u0aa1 \u0aaa\u0ac7\u0aa8\u0acd\u0ab8\u0abf\u0ab2 \u0aac\u0aa8\u0abe\u0ab5\u0ac0\u0a8f \u0a9b\u0ac0\u0a8f"),
        ("kn", "\u0ca8\u0cbe\u0cb5\u0cc1 \u0c97\u0ccd\u0cb0\u0cbe\u0cab\u0cc8\u0c9f\u0ccd \u0cb2\u0cc0\u0ca1\u0ccd \u0caa\u0cc6\u0ca8\u0ccd\u0cb8\u0cbf\u0cb2\u0ccd\u200c\u0c97\u0cb3\u0ca8\u0ccd\u0ca8\u0cc1 \u0ca4\u0caf\u0cbe\u0cb0\u0cbf\u0cb8\u0cc1\u0ca4\u0ccd\u0ca4\u0cc7\u0cb5\u0cc6"),
        ("ml", "\u0d1e\u0d19\u0d4d\u0d19\u0d7e \u0d17\u0d4d\u0d30\u0d3e\u0d2b\u0d48\u0d31\u0d4d\u0d31\u0d4d \u0d32\u0d40\u0d21\u0d4d \u0d2a\u0d46\u0d7b\u0d38\u0d3f\u0d32\u0d41\u0d15\u0d7e \u0d28\u0d3f\u0d7c\u0d2e\u0d4d\u0d2e\u0d3f\u0d15\u0d4d\u0d15\u0d41\u0d28\u0d4d\u0d28\u0d41"),
        ("pa", "\u0a05\u0a38\u0a40\u0a02 \u0a17\u0a4d\u0a30\u0a3e\u0a2b\u0a3e\u0a08\u0a1f \u0a32\u0a40\u0a21 \u0a2a\u0a48\u0a02\u0a38\u0a3f\u0a32\u0a3e\u0a02 \u0a2c\u0a23\u0a3e\u0a09\u0a02\u0a26\u0a47 \u0a39\u0a3e\u0a02"),
        ("ur", "\u06c1\u0645 \u06af\u0631\u06cc\u0641\u0627\u0626\u0679 \u0644\u06cc\u0688 \u067e\u0646\u0633\u0644 \u0628\u0646\u0627\u062a\u06d2 \u06c1\u06cc\u06ba"),
    ]

    for language, query in queries:
        response = client.post("/recommend", json={"query": query, "top_k": 5, "language": language})
        payload = response.json()

        assert response.status_code == 200
        assert payload["out_of_scope"] is True, language
        assert payload["retrieved_standards"] == [], language
        assert [item["code"] for item in payload["external_standards"]] == [
            "IS 1375:2021",
            "IS 2079:2022",
        ], language


def test_recommendations_use_bis_search_links_without_old_years():
    assert (
        _bis_portal_search_url_for_code("IS 1375:2021")
        == "https://standardsbis.bsbedge.com/BIS_SearchStandard.aspx?Standard_Number=IS+1375&id=0"
    )
    assert (
        _bis_portal_search_url_for_code("IS 404 (Part 1): 1993")
        == "https://standardsbis.bsbedge.com/BIS_SearchStandard.aspx?Standard_Number=IS+404+Part+1&id=0"
    )

    response = client.post("/recommend", json={"query": "IS 269:1989 ordinary portland cement", "top_k": 1})
    payload = response.json()

    assert response.status_code == 200
    assert payload["recommendations"][0]["source_url"] == (
        "https://standardsbis.bsbedge.com/BIS_SearchStandard.aspx?Standard_Number=IS+269&id=0"
    )
    assert "269_1989" not in payload["recommendations"][0]["source_url"]


def test_edible_oil_queries_return_relevant_oil_standards_with_links():
    queries = [
        ("en", "we make edible oil"),
        ("hi", "\u0939\u092e \u0916\u093e\u0926\u094d\u092f \u0924\u0947\u0932 \u092c\u0928\u093e\u0924\u0947 \u0939\u0948\u0902"),
        ("hinglish", "hum khane ka tel banate hain"),
        ("ta", "\u0ba8\u0bbe\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0b89\u0ba3\u0bb5\u0bc1 \u0b8e\u0ba3\u0bcd\u0ba3\u0bc6\u0baf\u0bcd \u0ba4\u0baf\u0bbe\u0bb0\u0bbf\u0b95\u0bcd\u0b95\u0bbf\u0bb1\u0bcb\u0bae\u0bcd"),
    ]

    for language, query in queries:
        response = client.post("/recommend", json={"query": query, "top_k": 5, "language": language})
        payload = response.json()

        assert response.status_code == 200
        assert payload["out_of_scope"] is True, language
        assert payload["retrieved_standards"] == [], language
        assert [item["code"] for item in payload["external_standards"]] == [
            "IS 548 (Part 1/Sec 1):2021",
            "IS 548 (Part 1/Sec 2):2021",
            "IS 548 (Part 2):1976",
            "IS 14349:2025",
            "IS 14636:1998",
        ], language
        assert all(item["source_url"].startswith("https://") for item in payload["external_standards"])
