import pytest
from fastapi.testclient import TestClient
from credit_scoring import app, check_sk_id_exists, fetch_data

client = TestClient(app)

# Tests for the /predict/{sk_id} endpoint
def test_predict_valid_sk_id():
    response = client.get("/predict/100002")
    assert response.status_code == 200
    data = response.json()
    assert "class" in data
    assert "probability_of_failure" in data
    #assert "prediction" in response.json()

def test_predict_invalid_sk_id():
    response = client.get("/predict/99999")
    assert response.status_code == 404
    assert response.json()["detail"] == "SK_ID_CURR not found"

def test_predict_missing_sk_id():
    response = client.get("/predict/")
    assert response.status_code == 404

# Tests for the auxiliary functions
def test_check_sk_id_exists_valid():
    assert check_sk_id_exists(100002) == True

def test_check_sk_id_exists_invalid():
    assert check_sk_id_exists(99999) == False

def test_fetch_data_valid():
    data = fetch_data(100002)
    assert data is not None

def test_fetch_data_invalid():
    data = fetch_data(99999)
    assert data.empty == True
