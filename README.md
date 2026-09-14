# mentor-analytics-backend

FastAPI service serving the CRM call-coverage dashboard. Reads directly from
a MongoDB Atlas database (`kylas`) via the async `motor` driver — no LLM
calls in the data path.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in MONGO_URI
uvicorn main:app --reload
```

## Endpoints

- `GET /api/health` — DB ping.
- `GET /api/owners` — CRM agents, for a filter dropdown.
- `GET /api/metrics?ownerId=<int>&start=<YYYY-MM-DD>&end=<YYYY-MM-DD>` — lead/call
  coverage KPIs for one owner over a date range.
