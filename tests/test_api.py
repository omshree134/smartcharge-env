from fastapi.testclient import TestClient

from service.api import app


client = TestClient(app)


def test_reset_state_and_step_endpoints():
    reset_response = client.post("/reset", json={"mode": "medium", "seed": 42})
    assert reset_response.status_code == 200
    reset_body = reset_response.json()
    assert reset_body["time_step"] == 0
    assert reset_body["slots_available"] == 3
    assert len(reset_body["vehicles"]) == 3

    state_response = client.get("/state")
    assert state_response.status_code == 200
    state_body = state_response.json()
    assert state_body["observation"] == reset_body
    assert state_body["task"]["id"] == "medium"
    assert state_body["progress"]["served"] == 0

    step_response = client.post("/step", json={"assignments": [0, 1, 2]})
    assert step_response.status_code == 200
    step_body = step_response.json()
    assert set(step_body.keys()) == {"observation", "reward", "done", "info"}
    assert 0.0 <= step_body["reward"] <= 1.0
    assert 0.0 <= step_body["info"]["score"] <= 1.0


def test_invalid_action_returns_400():
    client.post("/reset", json={"mode": "easy", "seed": 42})
    response = client.post("/step", json={"assignments": [3]})
    assert response.status_code == 400


def test_reset_accepts_missing_body():
    response = client.post("/reset")
    assert response.status_code == 200
    body = response.json()
    assert body["time_step"] == 0
