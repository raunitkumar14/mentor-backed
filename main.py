"""FastAPI backend for the CRM call-coverage dashboard.

Reads directly from the `kylas` MongoDB Atlas database via the async
`motor` driver. No LLM calls anywhere in the data path.
"""
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # must run before `db` reads MONGO_URI/MONGO_DB at import time

import db
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = db.connect()
    # Backs the {"name": {"$ne": None}} + sort("name") query in /api/owners —
    # without it, that sort falls back to an in-memory COLLSCAN sort that
    # gets slower as the users collection grows.
    await app.state.db.users.create_index("name")
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


_OWNERS_CACHE_TTL_SEC = 300
_owners_cache: dict = {"data": None, "expires_at": 0.0}


@app.get("/api/owners")
async def owners():
    # The owner roster changes rarely, but the frontend re-fetches it on
    # every page load — cache it briefly so repeated loads in one review
    # session don't each pay a DB round trip.
    now = time.monotonic()
    if _owners_cache["data"] is not None and now < _owners_cache["expires_at"]:
        return _owners_cache["data"]

    cursor = (
        app.state.db.users.find(
            {"name": {"$ne": None}},
            {"_id": 0, "id": 1, "name": 1},
        ).sort("name", 1)
    )
    data = await cursor.to_list(length=None)
    _owners_cache["data"] = data
    _owners_cache["expires_at"] = now + _OWNERS_CACHE_TTL_SEC
    return data


def _parse_owner_ids(raw: str) -> list[int]:
    try:
        ids = [int(part.strip()) for part in raw.split(",") if part.strip() != ""]
    except ValueError:
        raise HTTPException(
            status_code=400, detail="ownerIds must be a comma-separated list of integers"
        )
    if not ids:
        raise HTTPException(status_code=400, detail="ownerIds must not be empty")

    seen: set[int] = set()
    deduped = []
    for owner_id in ids:
        if owner_id not in seen:
            seen.add(owner_id)
            deduped.append(owner_id)
    return deduped


def _date_range(start: date, end: date) -> list[str]:
    days = []
    cursor = start
    while cursor <= end:
        days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return days


@app.get("/api/metrics")
async def metrics(
    ownerIds: str = Query(..., description="Comma-separated CRM owner ids"),
    start: date = Query(...),
    end: date = Query(...),
):
    owner_ids = _parse_owner_ids(ownerIds)
    if end < start:
        raise HTTPException(status_code=400, detail="end must be on or after start")

    start_iso = f"{start.isoformat()}T00:00:00.000Z"
    end_iso = f"{end.isoformat()}T23:59:59.999Z"

    pipeline = [
        {
            "$match": {
                "ownerId": {"$in": owner_ids},
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
                        "cond": {"$in": ["$$c.owner.id", owner_ids]},
                    }
                }
            }
        },
        {
            "$addFields": {
                "callCount": {"$size": "$calls"},
                "hasCalled": {"$gt": [{"$size": "$calls"}, 0]},
                # duration is in seconds and is null for calls that never
                # connected (no_answer/missed/rejected) — $sum over the
                # array skips those nulls rather than erroring.
                "callDurationSum": {"$sum": "$calls.duration"},
                "callsWithDuration": {
                    "$size": {
                        "$filter": {
                            "input": "$calls",
                            "as": "c",
                            "cond": {"$ne": ["$$c.duration", None]},
                        }
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
                            # Sum of every call attempt, not deduped by lead —
                            # this is what surfaces leads that got called
                            # more than once.
                            "totalCallAttempts": {"$sum": "$callCount"},
                            "maxCallsOnLead": {"$max": "$callCount"},
                            "totalCallDurationSec": {"$sum": "$callDurationSum"},
                            "callsWithDuration": {"$sum": "$callsWithDuration"},
                        }
                    }
                ],
                # Every individual call attempt's outcome — this sums to
                # totalCallAttempts, not leadsWithCalls, so a lead called
                # 3 times contributes 3 outcomes here, not 1.
                "outcomes": [
                    {"$unwind": "$calls"},
                    {
                        "$group": {
                            "_id": {"$ifNull": ["$calls.outcome", "unknown"]},
                            "count": {"$sum": 1},
                        }
                    },
                ],
                # Histogram of leads by how many times each was called —
                # this is what shows the "multiple calls for a single lead"
                # pattern instead of collapsing it into a single boolean.
                # Grouped by owner too, so the telecaller-wise table can
                # break the same histogram down per agent without a second
                # aggregation.
                "callDistribution": [
                    {
                        "$group": {
                            "_id": {
                                "ownerId": "$ownerId",
                                "bucket": {
                                    "$switch": {
                                        "branches": [
                                            {"case": {"$eq": ["$callCount", 0]}, "then": "0"},
                                            {"case": {"$eq": ["$callCount", 1]}, "then": "1"},
                                            {"case": {"$eq": ["$callCount", 2]}, "then": "2"},
                                        ],
                                        "default": "3+",
                                    }
                                },
                            },
                            "count": {"$sum": 1},
                        }
                    }
                ],
            }
        },
    ]

    [result] = await app.state.db.leads.aggregate(pipeline).to_list(length=1)

    totals = result["totals"][0] if result["totals"] else None
    total_leads = totals["totalLeads"] if totals else 0
    leads_with_calls = totals["leadsWithCalls"] if totals else 0
    total_call_attempts = totals["totalCallAttempts"] if totals else 0
    max_calls_on_lead = totals["maxCallsOnLead"] if totals else 0
    total_call_duration_sec = totals["totalCallDurationSec"] if totals else 0
    calls_with_duration = totals["callsWithDuration"] if totals else 0

    connect_rate_pct = (
        round(leads_with_calls / total_leads * 100, 1) if total_leads else 0
    )
    avg_calls_per_contacted_lead = (
        round(total_call_attempts / leads_with_calls, 2) if leads_with_calls else 0
    )
    avg_call_duration_sec = (
        round(total_call_duration_sec / calls_with_duration, 1)
        if calls_with_duration
        else 0
    )

    outcome_breakdown = {row["_id"]: row["count"] for row in result["outcomes"]}

    calls_per_lead_distribution = {"0": 0, "1": 0, "2": 0, "3+": 0}
    calls_per_lead_by_owner_map: dict[int, dict[str, int]] = {}
    for row in result["callDistribution"]:
        bucket = row["_id"]["bucket"]
        owner_id = row["_id"]["ownerId"]
        count = row["count"]
        calls_per_lead_distribution[bucket] += count
        calls_per_lead_by_owner_map.setdefault(owner_id, {})[bucket] = count

    calls_per_lead_distribution_by_owner = []
    for owner_id in owner_ids:
        counts = calls_per_lead_by_owner_map.get(owner_id, {})
        distribution = {b: counts.get(b, 0) for b in ("0", "1", "2", "3+")}
        calls_per_lead_distribution_by_owner.append(
            {
                "ownerId": owner_id,
                "distribution": distribution,
                "total": sum(distribution.values()),
            }
        )

    return {
        "totalLeads": total_leads,
        "leadsWithCalls": leads_with_calls,
        "leadsWithNoCall": total_leads - leads_with_calls,
        "connectRatePct": connect_rate_pct,
        "totalCallAttempts": total_call_attempts,
        "avgCallsPerContactedLead": avg_calls_per_contacted_lead,
        "maxCallsOnLead": max_calls_on_lead,
        "totalCallDurationSec": total_call_duration_sec,
        "avgCallDurationSec": avg_call_duration_sec,
        "callsPerLeadDistribution": calls_per_lead_distribution,
        "callsPerLeadDistributionByOwner": calls_per_lead_distribution_by_owner,
        "outcomeBreakdown": outcome_breakdown,
    }


@app.get("/api/lead-timeline")
async def lead_timeline(
    ownerIds: str = Query(..., description="Comma-separated CRM owner ids"),
    start: date = Query(...),
    end: date = Query(...),
):
    """Day-wise and owner-wise lead assignment, for the coverage graph.

    A lead belongs to exactly one owner, so summing across a multi-owner
    selection never double-counts a lead — no dedup step is needed beyond
    the `$in` match itself.
    """
    owner_ids = _parse_owner_ids(ownerIds)
    if end < start:
        raise HTTPException(status_code=400, detail="end must be on or after start")

    start_iso = f"{start.isoformat()}T00:00:00.000Z"
    end_iso = f"{end.isoformat()}T23:59:59.999Z"

    pipeline = [
        {
            "$match": {
                "ownerId": {"$in": owner_ids},
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
                        "cond": {"$in": ["$$c.owner.id", owner_ids]},
                    }
                }
            }
        },
        {
            "$addFields": {
                # "Connected" mirrors the KPI row's connect rate: at least
                # one call attempt exists on the lead, regardless of outcome.
                "hasCalled": {"$gt": [{"$size": "$calls"}, 0]},
                # createdAt is stored as an ISO-8601 string; its first 10
                # characters are the calendar date, cheaper than a real
                # $dateFromString/$dateToString round trip.
                "day": {"$substrCP": ["$createdAt", 0, 10]},
            }
        },
        {
            "$facet": {
                "byDay": [
                    {
                        "$group": {
                            "_id": "$day",
                            "total": {"$sum": 1},
                            "connected": {"$sum": {"$cond": ["$hasCalled", 1, 0]}},
                        }
                    }
                ],
                "byOwnerDay": [
                    {
                        "$group": {
                            "_id": {"ownerId": "$ownerId", "day": "$day"},
                            "count": {"$sum": 1},
                            "connected": {"$sum": {"$cond": ["$hasCalled", 1, 0]}},
                        }
                    }
                ],
            }
        },
    ]

    [result] = await app.state.db.leads.aggregate(pipeline).to_list(length=1)

    by_day = {row["_id"]: row for row in result["byDay"]}
    by_owner_day: dict[int, dict[str, dict]] = {owner_id: {} for owner_id in owner_ids}
    for row in result["byOwnerDay"]:
        by_owner_day.setdefault(row["_id"]["ownerId"], {})[row["_id"]["day"]] = row

    days = _date_range(start, end)

    day_wise = []
    for day in days:
        row = by_day.get(day)
        total = row["total"] if row else 0
        connected = row["connected"] if row else 0
        day_wise.append(
            {
                "date": day,
                "assigned": total,
                "connected": connected,
                "notConnected": total - connected,
            }
        )

    # "counts" (assigned) and "connected" are both keyed by day, so the
    # telecaller-wise table can show both figures per cell without a second
    # request.
    owner_wise = [
        {
            "ownerId": owner_id,
            "counts": {day: r["count"] for day, r in by_owner_day.get(owner_id, {}).items()},
            "connected": {
                day: r["connected"] for day, r in by_owner_day.get(owner_id, {}).items()
            },
        }
        for owner_id in owner_ids
    ]

    return {"days": days, "dayWise": day_wise, "ownerWise": owner_wise}


_OUTCOME_FIELD_MAP = {
    "connected": "connected",
    "no_answer": "noAnswer",
    "missed_call": "missedCall",
    "rejected": "rejected",
}


@app.get("/api/call-attempts-timeline")
async def call_attempts_timeline(
    ownerIds: str = Query(..., description="Comma-separated CRM owner ids"),
    start: date = Query(...),
    end: date = Query(...),
):
    """Day-wise attempted-call breakdowns, two ways, for the same total.

    Unlike /api/lead-timeline (which dates by lead assignment), this reads
    call_logs directly and dates by when the call itself happened — the
    natural axis for "how were calls answered" and "how often was the same
    lead re-called," both scoped to the calls made on that day.
    """
    owner_ids = _parse_owner_ids(ownerIds)
    if end < start:
        raise HTTPException(status_code=400, detail="end must be on or after start")

    start_iso = f"{start.isoformat()}T00:00:00.000Z"
    end_iso = f"{end.isoformat()}T23:59:59.999Z"

    pipeline = [
        {
            "$match": {
                "owner.id": {"$in": owner_ids},
                "createdAt": {"$gte": start_iso, "$lte": end_iso},
            }
        },
        {"$addFields": {"day": {"$substrCP": ["$createdAt", 0, 10]}}},
        {
            "$facet": {
                "outcomeByDay": [
                    {
                        "$group": {
                            "_id": {
                                "day": "$day",
                                "outcome": {"$ifNull": ["$outcome", "unknown"]},
                            },
                            "count": {"$sum": 1},
                        }
                    }
                ],
                # Bucketed by how many calls THIS lead got on THIS day, then
                # summed back to a call count (not a lead count) per bucket —
                # a lead called 3 times that day contributes all 3 to
                # "followup" (its first call ever that day already happened
                # before its second, so the whole day's contact history with
                # that lead reads as repeat outreach, not a first touch).
                "callsPerLeadByDay": [
                    {
                        "$group": {
                            "_id": {"day": "$day", "leadId": "$lead_id"},
                            "calls": {"$sum": 1},
                        }
                    },
                    {
                        "$addFields": {
                            "bucket": {
                                "$cond": [{"$eq": ["$calls", 1]}, "new", "followup"]
                            }
                        }
                    },
                    {
                        "$group": {
                            "_id": {"day": "$_id.day", "bucket": "$bucket"},
                            "attemptedCalls": {"$sum": "$calls"},
                        }
                    }
                ],
                # Same two breakdowns as above, collapsed over the date
                # range and split by owner instead — feeds the
                # telecaller-wise tables.
                "outcomeByOwner": [
                    {
                        "$group": {
                            "_id": {
                                "ownerId": "$owner.id",
                                "outcome": {"$ifNull": ["$outcome", "unknown"]},
                            },
                            "count": {"$sum": 1},
                        }
                    }
                ],
                "callsPerLeadByOwner": [
                    {
                        "$group": {
                            "_id": {"ownerId": "$owner.id", "leadId": "$lead_id"},
                            "calls": {"$sum": 1},
                        }
                    },
                    {
                        "$addFields": {
                            "bucket": {
                                "$cond": [{"$eq": ["$calls", 1]}, "new", "followup"]
                            }
                        }
                    },
                    {
                        "$group": {
                            "_id": {"ownerId": "$_id.ownerId", "bucket": "$bucket"},
                            "attemptedCalls": {"$sum": "$calls"},
                        }
                    },
                ],
            }
        },
    ]

    [result] = await app.state.db.call_logs.aggregate(pipeline).to_list(length=1)

    outcome_by_day: dict[str, dict[str, int]] = {}
    for row in result["outcomeByDay"]:
        day = row["_id"]["day"]
        outcome_by_day.setdefault(day, {})[row["_id"]["outcome"]] = row["count"]

    calls_per_lead_by_day: dict[str, dict[str, int]] = {}
    for row in result["callsPerLeadByDay"]:
        day = row["_id"]["day"]
        calls_per_lead_by_day.setdefault(day, {})[row["_id"]["bucket"]] = row[
            "attemptedCalls"
        ]

    days = _date_range(start, end)

    outcome_rows = []
    for day in days:
        counts = outcome_by_day.get(day, {})
        row = {"date": day, "total": sum(counts.values())}
        for raw_outcome, field in _OUTCOME_FIELD_MAP.items():
            row[field] = counts.get(raw_outcome, 0)
        outcome_rows.append(row)

    calls_per_lead_rows = []
    for day in days:
        counts = calls_per_lead_by_day.get(day, {})
        bucket_counts = {b: counts.get(b, 0) for b in ("new", "followup")}
        calls_per_lead_rows.append(
            {"date": day, "counts": bucket_counts, "total": sum(bucket_counts.values())}
        )

    outcome_by_owner_map: dict[int, dict[str, int]] = {}
    for row in result["outcomeByOwner"]:
        outcome_by_owner_map.setdefault(row["_id"]["ownerId"], {})[row["_id"]["outcome"]] = row[
            "count"
        ]

    outcome_by_owner_rows = []
    for owner_id in owner_ids:
        counts = outcome_by_owner_map.get(owner_id, {})
        row = {"ownerId": owner_id, "total": sum(counts.values())}
        for raw_outcome, field in _OUTCOME_FIELD_MAP.items():
            row[field] = counts.get(raw_outcome, 0)
        outcome_by_owner_rows.append(row)

    calls_per_lead_by_owner_map: dict[int, dict[str, int]] = {}
    for row in result["callsPerLeadByOwner"]:
        calls_per_lead_by_owner_map.setdefault(row["_id"]["ownerId"], {})[
            row["_id"]["bucket"]
        ] = row["attemptedCalls"]

    calls_per_lead_by_owner_rows = []
    for owner_id in owner_ids:
        counts = calls_per_lead_by_owner_map.get(owner_id, {})
        bucket_counts = {b: counts.get(b, 0) for b in ("new", "followup")}
        calls_per_lead_by_owner_rows.append(
            {"ownerId": owner_id, "counts": bucket_counts, "total": sum(bucket_counts.values())}
        )

    return {
        "days": days,
        "outcomeByDay": outcome_rows,
        "callsPerLeadByDay": calls_per_lead_rows,
        "outcomeByOwner": outcome_by_owner_rows,
        "callsPerLeadByOwner": calls_per_lead_by_owner_rows,
    }


_DURATION_BUCKETS = ["0-1", "1-2", "2-3", "3-4", "4-5", "5+"]


@app.get("/api/call-analytics")
async def call_analytics(
    ownerIds: str = Query(..., description="Comma-separated CRM owner ids"),
    start: date = Query(...),
    end: date = Query(...),
):
    """Two aggregate readings of the same filtered call set.

    - connectRateByOwnerDate: per owner, per day, connected/total*100 (null
      when that owner made no calls that day — never divide by zero).
    - durationDistribution: fixed 1-minute buckets over calls that actually
      have a duration (a call that never connected has none, so it can't be
      "0-1 min" — it simply isn't in any bucket).
    """
    owner_ids = _parse_owner_ids(ownerIds)
    if end < start:
        raise HTTPException(status_code=400, detail="end must be on or after start")

    start_iso = f"{start.isoformat()}T00:00:00.000Z"
    end_iso = f"{end.isoformat()}T23:59:59.999Z"

    pipeline = [
        {
            "$match": {
                "owner.id": {"$in": owner_ids},
                "createdAt": {"$gte": start_iso, "$lte": end_iso},
            }
        },
        {
            "$facet": {
                "outcomeByOwnerDay": [
                    {"$addFields": {"day": {"$substrCP": ["$createdAt", 0, 10]}}},
                    {
                        "$group": {
                            "_id": {"ownerId": "$owner.id", "day": "$day"},
                            "total": {"$sum": 1},
                            "connected": {
                                "$sum": {"$cond": [{"$eq": ["$outcome", "connected"]}, 1, 0]}
                            },
                        }
                    },
                ],
                # Grouped by owner too (in addition to bucket), so the
                # telecaller-wise table is one aggregation, not a second
                # query — the day-agnostic overall distribution below is
                # just this same result summed back across owners.
                "durationCounts": [
                    {"$match": {"duration": {"$ne": None}}},
                    {
                        "$addFields": {
                            "bucket": {
                                "$switch": {
                                    "branches": [
                                        {"case": {"$lt": ["$duration", 60]}, "then": "0-1"},
                                        {"case": {"$lt": ["$duration", 120]}, "then": "1-2"},
                                        {"case": {"$lt": ["$duration", 180]}, "then": "2-3"},
                                        {"case": {"$lt": ["$duration", 240]}, "then": "3-4"},
                                        {"case": {"$lt": ["$duration", 300]}, "then": "4-5"},
                                    ],
                                    "default": "5+",
                                }
                            }
                        }
                    },
                    {
                        "$group": {
                            "_id": {"ownerId": "$owner.id", "bucket": "$bucket"},
                            "count": {"$sum": 1},
                        }
                    },
                ],
            }
        },
    ]

    [result] = await app.state.db.call_logs.aggregate(pipeline).to_list(length=1)

    by_owner_day: dict[int, dict[str, dict]] = {owner_id: {} for owner_id in owner_ids}
    for row in result["outcomeByOwnerDay"]:
        owner_id = row["_id"]["ownerId"]
        by_owner_day.setdefault(owner_id, {})[row["_id"]["day"]] = row

    days = _date_range(start, end)

    # Per-owner and grand totals, aggregated from the same per-day cells
    # the rates above are computed from — a true volume-weighted rate, not
    # an average of daily percentages (which would misweight low-volume days).
    connect_rate_rows = []
    owner_totals = []
    overall_attempted = 0
    overall_connected = 0
    for owner_id in owner_ids:
        rates = {}
        owner_attempted = 0
        owner_connected = 0
        for day in days:
            cell = by_owner_day.get(owner_id, {}).get(day)
            if not cell or cell["total"] == 0:
                rates[day] = None
            else:
                rates[day] = round(cell["connected"] / cell["total"] * 100, 1)
                owner_attempted += cell["total"]
                owner_connected += cell["connected"]
        connect_rate_rows.append({"ownerId": owner_id, "rates": rates})
        owner_totals.append(
            {
                "ownerId": owner_id,
                "totalAttempted": owner_attempted,
                "totalConnected": owner_connected,
                "connectRatePct": (
                    round(owner_connected / owner_attempted * 100, 1)
                    if owner_attempted
                    else None
                ),
            }
        )
        overall_attempted += owner_attempted
        overall_connected += owner_connected

    duration_counts: dict[str, int] = {b: 0 for b in _DURATION_BUCKETS}
    duration_by_owner_map: dict[int, dict[str, int]] = {}
    for row in result["durationCounts"]:
        bucket = row["_id"]["bucket"]
        owner_id = row["_id"]["ownerId"]
        count = row["count"]
        duration_counts[bucket] += count
        duration_by_owner_map.setdefault(owner_id, {})[bucket] = count

    duration_distribution = [
        {"bucket": b, "count": duration_counts.get(b, 0)} for b in _DURATION_BUCKETS
    ]

    duration_by_owner = []
    for owner_id in owner_ids:
        counts = duration_by_owner_map.get(owner_id, {})
        buckets = {b: counts.get(b, 0) for b in _DURATION_BUCKETS}
        duration_by_owner.append(
            {"ownerId": owner_id, "buckets": buckets, "total": sum(buckets.values())}
        )

    return {
        "days": days,
        "connectRateByOwnerDate": connect_rate_rows,
        "connectRateOwnerTotals": owner_totals,
        "connectRateOverall": {
            "totalAttempted": overall_attempted,
            "totalConnected": overall_connected,
            "connectRatePct": (
                round(overall_connected / overall_attempted * 100, 1)
                if overall_attempted
                else None
            ),
        },
        "durationDistribution": duration_distribution,
        "durationByOwner": duration_by_owner,
    }
