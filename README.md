## NoShowIQ (MLOps Midterm — Spring 2026)

Predicts the risk of a patient not showing up for a medical appointment.

### Status
- CI badge: [![ci-cd](https://github.com/rajazain2001/noshow-iq-23769/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/rajazain2001/noshow-iq-23769/actions/workflows/ci-cd.yml)
- **Live URL**: _TBD (Hugging Face Spaces)_
- **Docker image**: https://hub.docker.com/r/<DOCKER_USERNAME>/noshow-iq-23769

### Local development (day 1)
Create and activate a virtual environment, then install:

```bash
pip install -r requirements.txt
```

Dataset expected at `data/raw/KaggleV2-May-2016.csv`.

### Run locally (Docker)
```bash
docker compose up --build
```

Open:
- UI: http://localhost:8000/
- Docs: http://localhost:8000/docs
- mongo-express: http://localhost:8081

### Smoke test (after deployment)
```bash
python smoke_test.py https://<your-space>.hf.space
```

