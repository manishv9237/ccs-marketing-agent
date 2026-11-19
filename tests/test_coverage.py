from agent import coverage


def test_coverage_counts_and_status():
    chunks = ["We launched a new range for Diwali with a big discount."] * 12 + ["Our ESG report covers carbon and renewable energy."] * 3 + ["Plain text."]
    rows = {r["Goal"]: r for r in coverage.coverage(chunks)}
    assert rows["Launch"]["Chunks"] == 12 and rows["Launch"]["Status"] == "Adequate"
    assert rows["Festive offer"]["Chunks"] == 12
    assert rows["Sustainability"]["Chunks"] == 3 and rows["Sustainability"]["Status"] == "Thin"
    assert rows["Hiring"]["Chunks"] == 0 and coverage.status(30) == "Good"


def test_service_goal_coverage(settings):
    from agent.embeddings import HashEmbedder
    from agent.service import Service
    svc = Service(settings, llm_fn=lambda *a, **k: "{}", embedder=HashEmbedder())
    b = svc.create_brand("u1", "Acme", "FMCG & Consumer Brands")
    assert [r["Goal"] for r in svc.goal_coverage("u1", b["brand_id"])] == ["Launch", "Festive offer", "Hiring", "Sustainability"]
