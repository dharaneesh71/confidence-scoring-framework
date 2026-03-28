"""
Test configuration â€” patches out heavy ML services so pytest
never triggers a model download or ChromaDB init.
"""
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_heavy_services():
    """
    FIX #14: Automatically applied to every test.
    Prevents LlamaService, ChromaService, and ScoringService
    from loading real models during the test run.
    """
    mock_llama   = MagicMock()
    mock_chroma  = MagicMock()
    mock_scoring = MagicMock()

    mock_llama.is_ready.return_value   = True
    mock_chroma.is_ready.return_value  = True
    mock_chroma.get_count.return_value = 0
    mock_chroma.search.return_value    = []
    mock_llama.generate_answer.return_value = "Mock answer"
    mock_scoring.compute_confidence_score.return_value = (0.9, "Mock explanation", [], {})

    with patch("api.routes.endpoints.llama_service",   mock_llama), \
         patch("api.routes.endpoints.chroma_service",  mock_chroma), \
         patch("api.routes.endpoints.scoring_service", mock_scoring):
        yield