from uuid import uuid4

from fastapi.testclient import TestClient

from app.database.session import SessionLocal
from app.main import app
from app.models.user import User


client = TestClient(app)


def unique_credentials():
    suffix = uuid4().hex[:10]
    return {
        "username": f"testuser_{suffix}",
        "email": f"{suffix}@example.com",
        "password": "TestPassword123!",
    }


def delete_user(username: str):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user is not None:
            db.delete(user)
            db.commit()
    finally:
        db.close()


def test_register_login_and_get_me():
    credentials = unique_credentials()

    try:
        register_response = client.post(
            "/auth/register",
            json=credentials,
        )

        assert register_response.status_code == 201
        registered = register_response.json()
        assert registered["username"] == credentials["username"]
        assert registered["email"] == credentials["email"]
        assert "hashed_password" not in registered

        login_response = client.post(
            "/auth/login",
            data={
                "username": credentials["username"],
                "password": credentials["password"],
            },
        )

        assert login_response.status_code == 200
        token_data = login_response.json()
        assert token_data["token_type"] == "bearer"
        assert token_data["access_token"]

        me_response = client.get(
            "/auth/me",
            headers={
                "Authorization": f"Bearer {token_data['access_token']}",
            },
        )

        assert me_response.status_code == 200
        assert me_response.json()["username"] == credentials["username"]
    finally:
        delete_user(credentials["username"])


def test_register_duplicate_username():
    credentials = unique_credentials()

    try:
        first = client.post("/auth/register", json=credentials)
        assert first.status_code == 201

        duplicate = client.post(
            "/auth/register",
            json={
                **credentials,
                "email": f"other_{credentials['email']}",
            },
        )

        assert duplicate.status_code == 409
        assert "Username already exists" in duplicate.json()["detail"]
    finally:
        delete_user(credentials["username"])


def test_register_duplicate_email():
    first_credentials = unique_credentials()
    second_credentials = unique_credentials()

    try:
        first = client.post(
            "/auth/register",
            json=first_credentials,
        )
        assert first.status_code == 201

        duplicate = client.post(
            "/auth/register",
            json={
                **second_credentials,
                "email": first_credentials["email"],
            },
        )

        assert duplicate.status_code == 409
        assert "Email already exists" in duplicate.json()["detail"]
    finally:
        delete_user(first_credentials["username"])
        delete_user(second_credentials["username"])


def test_login_invalid_password():
    credentials = unique_credentials()

    try:
        response = client.post(
            "/auth/register",
            json=credentials,
        )
        assert response.status_code == 201

        login_response = client.post(
            "/auth/login",
            data={
                "username": credentials["username"],
                "password": "WrongPassword123!",
            },
        )

        assert login_response.status_code == 401
    finally:
        delete_user(credentials["username"])


def test_me_requires_authentication():
    response = client.get("/auth/me")

    assert response.status_code == 401


def test_register_rejects_short_password():
    credentials = unique_credentials()
    credentials["password"] = "short"

    response = client.post(
        "/auth/register",
        json=credentials,
    )

    assert response.status_code == 422
