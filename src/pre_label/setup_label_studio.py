"""
透過 Label Studio API 自動建立專案並匯入預標註資料。
前提：Label Studio 已在 http://localhost:8080 執行。

Usage:
    python -m src.pre_label.setup_label_studio
"""

import json
import os
import requests

LS_URL = "http://localhost:8080"

# ESG 9-category labeling config
LABEL_CONFIG = """\
<View>
  <Header value="ESG Classification" />
  <Text name="text" value="$text" />
  <Header value="Paragraph Info" size="4" />
  <View style="display: flex; gap: 10px; margin-bottom: 10px;">
    <Label value="ID: $paragraph_id" />
    <Label value="Ticker: $ticker" />
    <Label value="Year: $year" />
  </View>
  <Choices name="label" toName="text" choice="single" showInline="true">
    <Choice value="Climate Change" />
    <Choice value="Natural Capital" />
    <Choice value="Pollution &amp; Waste" />
    <Choice value="Human Capital" />
    <Choice value="Product Liability" />
    <Choice value="Community Relations" />
    <Choice value="Corporate Governance" />
    <Choice value="Business Ethics &amp; Values" />
    <Choice value="Non-ESG" />
  </Choices>
</View>
"""


def get_api_token(email: str, password: str) -> str:
    """Login and get API token."""
    # Try signup first (first time), then login
    for endpoint in ["/user/signup", "/user/login"]:
        resp = requests.post(
            f"{LS_URL}{endpoint}",
            json={"email": email, "password": password},
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code == 200 and "token" in resp.text:
            break

    # Get token from /api/current-user/token
    session = requests.Session()
    session.post(
        f"{LS_URL}/user/login",
        data={"email": email, "password": password},
    )
    resp = session.get(f"{LS_URL}/api/current-user/token")
    if resp.status_code == 200:
        return resp.json().get("token", "")

    raise RuntimeError(f"Cannot get API token. Status: {resp.status_code}, Body: {resp.text}")


def create_project(token: str) -> int:
    """Create ESG Classification project, return project ID."""
    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    resp = requests.post(
        f"{LS_URL}/api/projects",
        headers=headers,
        json={
            "title": "ESG Classification",
            "description": "ESG 9-category classification of 10-K Item 1A paragraphs",
            "label_config": LABEL_CONFIG,
        },
    )
    resp.raise_for_status()
    project_id = resp.json()["id"]
    print(f"[OK] Project created: ID={project_id}")
    return project_id


def import_data(token: str, project_id: int, data_path: str):
    """Import pre-annotated data into the project."""
    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}

    with open(data_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    print(f"Importing {len(tasks)} tasks...")
    resp = requests.post(
        f"{LS_URL}/api/projects/{project_id}/import",
        headers=headers,
        json=tasks,
    )
    resp.raise_for_status()
    result = resp.json()
    print(f"[OK] Imported: {result.get('task_count', len(tasks))} tasks")


def main():
    email = "admin@esg.local"
    password = "admin123"
    data_path = "data/10k_1A/label_studio_import.json"

    print("1. Getting API token...")
    token = get_api_token(email, password)
    print(f"   Token: {token[:10]}...")

    print("2. Creating project...")
    project_id = create_project(token)

    print("3. Importing data...")
    import_data(token, project_id, data_path)

    print()
    print("=" * 50)
    print(f"Label Studio is ready!")
    print(f"  URL:     {LS_URL}/projects/{project_id}")
    print(f"  Email:   {email}")
    print(f"  Password: {password}")
    print(f"  Tasks:   679 paragraphs with ESG pre-annotations")
    print("=" * 50)


if __name__ == "__main__":
    main()
