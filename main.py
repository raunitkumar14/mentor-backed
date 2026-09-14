"""FastAPI backend for the CRM call-coverage dashboard.

Reads directly from the `kylas` MongoDB Atlas database via the async
`motor` driver. No LLM calls anywhere in the data path.
"""
from contextlib import asynccontextmanager
from datetime import date
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # must run before `db` reads MONGO_URI/MONGO_DB at import time

import db
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = db.connect()
    yield
    db.close()


app = FastAPI(title="CRM Call-Coverage Dashboard API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    await app.state.db.command("ping")
    return {"status": "ok"}


@app.get("/api/owners")
async def owners():
    cursor = (
        app.state.db.users.find(
            {"name": {"$ne": None}},
            {"_id": 0, "id": 1, "name": 1},
        ).sort("name", 1)
    )
    return await cursor.to_list(length=None)


@app.get("/api/metrics")
async def metrics(
    ownerId: int = Query(...),
    start: date = Query(...),
    end: date = Query(...),
):
    if end < start:
        raise HTTPException(status_code=400, detail="end must be on or after start")

    start_iso = f"{start.isoformat()}T00:00:00.000Z"
    end_iso = f"{end.isoformat()}T23:59:59.999Z"

    pipeline = [
        {
            "$match": {
                "ownerId": ownerId,
                "createdAt": {"$gte": start_iso, "$lte": end_iso},
            }
        },
        {
            "$lookup": {
                "from": "call_logs",
                "localField": "id",
                "foreignField": "lead_id",
                "as": "calls",
            }
        },
        {
            "$addFields": {
                "calls": {
                    "$filter": {
                        "input": "$calls",
                        "as": "c",
                        "cond": {"$eq": ["$$c.owner.id", ownerId]},
                    }
                }
            }
        },
        {
            "$addFields": {
                "hasCalled": {"$gt": [{"$size": "$calls"}, 0]},
                "latestOutcome": {
                    "$let": {
                        "vars": {
                            "sortedCalls": {
                                "$sortArray": {
                                    "input": "$calls",
                                    "sortBy": {"createdAt": -1},
                                }
                            }
                        },
                        "in": {"$arrayElemAt": ["$$sortedCalls.outcome", 0]},
                    }
                },
            }
        },
        {
            "$facet": {
                "totals": [
                    {
                        "$group": {
                            "_id": None,
                            "totalLeads": {"$sum": 1},
                            "leadsWithCalls": {
                                "$sum": {"$cond": ["$hasCalled", 1, 0]}
                            },
                        }
                    }
                ],
                "outcomes": [
                    {"$match": {"hasCalled": True}},
                    {
                        "$group": {
                            "_id": {"$ifNull": ["$latestOutcome", "unknown"]},
                            "count": {"$sum": 1},
                        }
                    },
                ],
            }
        },
    ]

    [result] = await app.state.db.leads.aggregate(pipeline).to_list(length=1)

    totals = result["totals"][0] if result["totals"] else None
    total_leads = totals["totalLeads"] if totals else 0
    leads_with_calls = totals["leadsWithCalls"] if totals else 0

    connect_rate_pct = (
        round(leads_with_calls / total_leads * 100, 1) if total_leads else 0
    )

    outcome_breakdown = {row["_id"]: row["count"] for row in result["outcomes"]}

    return {
        "totalLeads": total_leads,
        "leadsWithCalls": leads_with_calls,
        "leadsWithNoCall": total_leads - leads_with_calls,
        "connectRatePct": connect_rate_pct,
        "outcomeBreakdown": outcome_breakdown,
    }
