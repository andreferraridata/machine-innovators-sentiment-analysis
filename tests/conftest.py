import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    # Context manager runs the lifespan: the model is loaded once for the whole session.
    with TestClient(app) as c:
        yield c
