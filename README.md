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
- `GET /api/metrics?ownerIds=<int,int,...>&start=<YYYY-MM-DD>&end=<YYYY-MM-DD>` —
  lead/call coverage KPIs for one or more owners (summed) over a date range.
- `GET /api/lead-timeline?ownerIds=<int,int,...>&start=<YYYY-MM-DD>&end=<YYYY-MM-DD>` —
  day-by-day assigned/connected/not-connected lead counts (summed across the
  selected owners) plus a per-owner, per-day assignment breakdown, for the
  lead assignment & connection graph.
- `GET /api/call-attempts-timeline?ownerIds=<int,int,...>&start=<YYYY-MM-DD>&end=<YYYY-MM-DD>` —
  day-by-day attempted-call breakdowns, dated by the call itself (not the
  lead's assignment date): by outcome (connected/no-answer/missed/rejected)
  and by how many calls the same lead got that day (1/2/3/4+, valued in call
  counts, not lead counts) — two slices of the same daily total.
- `GET /api/call-analytics?ownerIds=<int,int,...>&start=<YYYY-MM-DD>&end=<YYYY-MM-DD>` —
  two aggregate readings of the same filtered call set: a per-owner/per-day
  connected-call rate (null when that owner made no calls that day), and a
  fixed-bucket call-duration distribution (only over calls with a recorded
  duration — a call that never connected has none and isn't counted in any
  bucket).
