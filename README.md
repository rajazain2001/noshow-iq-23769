## NoShowIQ (MLOps Midterm — Spring 2026)

Predicts the risk of a patient not showing up for a medical appointment.

### Status
- CI badge: [![ci-cd](https://github.com/rajazain2001/noshow-iq-23769/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/rajazain2001/noshow-iq-23769/actions/workflows/ci-cd.yml)
- **Live URL**: https://raja2001-noshow-iq-23769.hf.space
- **Docker image**: https://hub.docker.com/r/raja1zain/noshow-iq-23769

### Local development (day 1)
Create and activate a virtual environment, then install:

```bash
pip install -r requirements.txt
```

Dataset expected at `data/raw/KaggleV2-May-2016.csv`.

### Smoke test (after deployment)
Run:
```bash
python smoke_test.py https://<your-space>.hf.space
```

