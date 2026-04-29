from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set. In PowerShell run:\n"
            '$env:HF_TOKEN="hf_..."\n'
            "then re-run this script."
        )

    repo_id = os.getenv("HF_SPACE_REPO", "raja2001/noshow-iq-23769")
    local_html = Path(__file__).resolve().parents[1] / "noshow_iq" / "static" / "noshow_iq_dashboard.html"
    if not local_html.exists():
        raise SystemExit(f"Dashboard not found at {local_html}")

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(local_html),
        path_in_repo="noshow_iq/static/noshow_iq_dashboard.html",
        repo_id=repo_id,
        repo_type="space",
        commit_message="refactor(ui): simplify dashboard and auto-connect",
    )
    print(f"Uploaded dashboard to Space: {repo_id}")


if __name__ == "__main__":
    main()

