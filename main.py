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
    # Backs the per-call "is this the lead's all-time first call" $lookup in
    # /api/call-attempts-timeline — that lookup sorts by createdAt within a
    # lead_id match, so this composite index turns it into an index-only
    # min lookup instead of a per-lead COLLSCAN.
    await app.state.db.call_logs.create_index([("lead_id", 1), ("createdAt", 1)])
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
                # True only if at least one attempt actually got through —
                # distinct from hasCalled, which is true even when every
                # attempt was a no-answer/missed/rejected.
                "hasConnected": {
                    "$gt": [
                        {
                            "$size": {
                                "$filter": {
                                    "input": "$calls",
                                    "as": "c",
                                    "cond": {"$eq": ["$$c.outcome", "connected"]},
                                }
                            }
                        },
                        0,
                    ]
                },
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
                            "leadsAttempted": {
                                "$sum": {"$cond": ["$hasCalled", 1, 0]}
                            },
                            "leadsConnected": {
                                "$sum": {"$cond": ["$hasConnected", 1, 0]}
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
                # totalCallAttempts, not leadsAttempted, so a lead called
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
    leads_attempted = totals["leadsAttempted"] if totals else 0
    leads_connected = totals["leadsConnected"] if totals else 0
    total_call_attempts = totals["totalCallAttempts"] if totals else 0
    max_calls_on_lead = totals["maxCallsOnLead"] if totals else 0
    total_call_duration_sec = totals["totalCallDurationSec"] if totals else 0
    calls_with_duration = totals["callsWithDuration"] if totals else 0

    # Attempt rate: leads that got at least one call, connected or not, out
    # of every lead in range. Connect rate: of the leads actually attempted,
    # how many connected — an unattempted lead couldn't have connected, so
    # it belongs out of this denominator (matches how every other connect
    # rate in this API, e.g. /api/call-analytics, is computed).
    attempt_rate_pct = (
        round(leads_attempted / total_leads * 100, 1) if total_leads else 0
    )
    connect_rate_pct = (
        round(leads_connected / leads_attempted * 100, 1) if leads_attempted else 0
    )
    avg_calls_per_contacted_lead = (
        round(total_call_attempts / leads_attempted, 2) if leads_attempted else 0
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
        "leadsNotAttempted": total_leads - leads_attempted,
        "leadsAttempted": leads_attempted,
        "leadsConnected": leads_connected,
        "leadsUnconnected": leads_attempted - leads_connected,
        "attemptRatePct": attempt_rate_pct,
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

# Incoming and outgoing calls use different outcome vocabularies in the CRM
# (incoming: connected/missed_call/no_answer/rejected; outgoing:
# connected/no_answer only), so each direction gets its own bucket
# expression, shared by every endpoint that splits calls by direction.
# Incoming's rare "no_answer" (distinct from "missed_call") folds into
# "missed" so incoming keeps exactly three buckets without dropping calls.
_INCOMING_BUCKET_EXPR = {
    "$switch": {
        "branches": [
            {"case": {"$eq": ["$outcome", "connected"]}, "then": "connected"},
            {"case": {"$eq": ["$outcome", "rejected"]}, "then": "rejected"},
        ],
        "default": "missed",
    }
}
_OUTGOING_BUCKET_EXPR = {"$cond": [{"$eq": ["$outcome", "connected"]}, "connected", "noAnswer"]}
_INCOMING_BUCKETS = ["connected", "missed", "rejected"]
_OUTGOING_BUCKETS = ["connected", "noAnswer"]


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
        # "todayLead" vs "olderLead" buckets by the LEAD's creation date, not
        # the call's — distinct from the "new"/"followup" buckets below,
        # which are about first-ever contact with the lead, not same-day
        # timing. A call can be "followup" (the lead already had a call
        # before, possibly by another owner, possibly before this date
        # range) and still be on "todayLead" (the lead was also created
        # today), so the two axes are independent.
        {
            "$lookup": {
                "from": "leads",
                "localField": "lead_id",
                "foreignField": "id",
                "as": "lead",
            }
        },
        {
            "$addFields": {
                "leadCreatedDay": {
                    "$substrCP": [
                        {"$ifNull": [{"$arrayElemAt": ["$lead.createdAt", 0]}, ""]},
                        0,
                        10,
                    ]
                }
            }
        },
        {
            "$addFields": {
                "leadAgeBucket": {
                    "$cond": [
                        {"$eq": ["$leadCreatedDay", "$day"]},
                        "todayLead",
                        "olderLead",
                    ]
                }
            }
        },
        # "new" vs "followup" is decided against the lead's ENTIRE call
        # history (any owner, any date — not just this filtered range): a
        # call is "new" only if it's the single earliest call_logs document
        # ever recorded for that lead_id. Every other call to that lead,
        # same day or ten days later, same owner or a different one, is
        # "followup". Tie-broken by _id so a duplicate-timestamp import
        # can't mint two "new" calls for one lead.
        {
            "$lookup": {
                "from": "call_logs",
                "let": {"leadId": "$lead_id"},
                "pipeline": [
                    {"$match": {"$expr": {"$eq": ["$lead_id", "$$leadId"]}}},
                    {"$sort": {"createdAt": 1, "_id": 1}},
                    {"$limit": 1},
                    {"$project": {"_id": 1}},
                ],
                "as": "firstCallForLead",
            }
        },
        {
            "$addFields": {
                "callBucket": {
                    "$cond": [
                        {"$eq": ["$_id", {"$arrayElemAt": ["$firstCallForLead._id", 0]}]},
                        "new",
                        "followup",
                    ]
                }
            }
        },
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
                "callsPerLeadByDay": [
                    {
                        "$group": {
                            "_id": {"day": "$day", "bucket": "$callBucket"},
                            "attemptedCalls": {"$sum": 1},
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
                            "_id": {"ownerId": "$owner.id", "bucket": "$callBucket"},
                            "attemptedCalls": {"$sum": 1},
                        }
                    },
                ],
                "leadAgeByDay": [
                    {
                        "$group": {
                            "_id": {"day": "$day", "bucket": "$leadAgeBucket"},
                            "count": {"$sum": 1},
                        }
                    }
                ],
                # Distinct leads actually called in this range (not the same
                # as the "new" bucket above, which only counts leads whose
                # very first-ever call fell in this range) — the denominator
                # for "average calls per lead".
                "distinctLeads": [
                    {"$group": {"_id": "$lead_id"}},
                    {"$count": "count"},
                ],
                "distinctLeadsByOwner": [
                    {"$group": {"_id": {"ownerId": "$owner.id", "leadId": "$lead_id"}}},
                    {"$group": {"_id": "$_id.ownerId", "count": {"$sum": 1}}},
                ],
                # Feeds the date-wise outcome table: total logged duration
                # and distinct-lead count per day (duration is only ever
                # set on a call that connected, so this $match isn't a
                # filter so much as skipping calls that have none to add).
                "durationByDay": [
                    {"$match": {"duration": {"$ne": None}}},
                    {"$group": {"_id": "$day", "totalDuration": {"$sum": "$duration"}}},
                ],
                "distinctLeadsByDay": [
                    {"$group": {"_id": {"day": "$day", "leadId": "$lead_id"}}},
                    {"$group": {"_id": "$_id.day", "count": {"$sum": 1}}},
                ],
                "incomingOutcomeByDay": [
                    {"$match": {"callType": "incoming"}},
                    {"$addFields": {"bucket": _INCOMING_BUCKET_EXPR}},
                    {
                        "$group": {
                            "_id": {"day": "$day", "bucket": "$bucket"},
                            "count": {"$sum": 1},
                        }
                    },
                ],
                "outgoingOutcomeByDay": [
                    {"$match": {"callType": "outgoing"}},
                    {"$addFields": {"bucket": _OUTGOING_BUCKET_EXPR}},
                    {
                        "$group": {
                            "_id": {"day": "$day", "bucket": "$bucket"},
                            "count": {"$sum": 1},
                        }
                    },
                ],
                # Same three, collapsed over the date range and split by
                # owner instead — feeds the telecaller-wise outcome table.
                "durationByOwner": [
                    {"$match": {"duration": {"$ne": None}}},
                    {"$group": {"_id": "$owner.id", "totalDuration": {"$sum": "$duration"}}},
                ],
                "incomingOutcomeByOwner": [
                    {"$match": {"callType": "incoming"}},
                    {"$addFields": {"bucket": _INCOMING_BUCKET_EXPR}},
                    {
                        "$group": {
                            "_id": {"ownerId": "$owner.id", "bucket": "$bucket"},
                            "count": {"$sum": 1},
                        }
                    },
                ],
                "outgoingOutcomeByOwner": [
                    {"$match": {"callType": "outgoing"}},
                    {"$addFields": {"bucket": _OUTGOING_BUCKET_EXPR}},
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

    lead_age_by_day: dict[str, dict[str, int]] = {}
    for row in result["leadAgeByDay"]:
        day = row["_id"]["day"]
        lead_age_by_day.setdefault(day, {})[row["_id"]["bucket"]] = row["count"]

    lead_age_rows = []
    for day in days:
        counts = lead_age_by_day.get(day, {})
        bucket_counts = {b: counts.get(b, 0) for b in ("todayLead", "olderLead")}
        lead_age_rows.append(
            {"date": day, "counts": bucket_counts, "total": sum(bucket_counts.values())}
        )

    distinct_leads = (
        result["distinctLeads"][0]["count"] if result["distinctLeads"] else 0
    )
    distinct_leads_by_owner_map = {
        row["_id"]: row["count"] for row in result["distinctLeadsByOwner"]
    }
    distinct_leads_by_owner_rows = [
        {"ownerId": owner_id, "count": distinct_leads_by_owner_map.get(owner_id, 0)}
        for owner_id in owner_ids
    ]

    duration_by_day = {row["_id"]: row["totalDuration"] for row in result["durationByDay"]}
    distinct_leads_by_day = {row["_id"]: row["count"] for row in result["distinctLeadsByDay"]}

    incoming_by_day: dict[str, dict[str, int]] = {}
    for row in result["incomingOutcomeByDay"]:
        incoming_by_day.setdefault(row["_id"]["day"], {})[row["_id"]["bucket"]] = row["count"]

    outgoing_by_day: dict[str, dict[str, int]] = {}
    for row in result["outgoingOutcomeByDay"]:
        outgoing_by_day.setdefault(row["_id"]["day"], {})[row["_id"]["bucket"]] = row["count"]

    # Date-wise table backing the Outcome tab: duration and lead-coverage
    # stats alongside the incoming/outgoing outcome split for the same day
    # — a manager reading one row sees both "how much talk time" and "how
    # did those calls resolve" without cross-referencing two charts.
    outcome_table = []
    for day in days:
        total_duration = duration_by_day.get(day, 0)
        leads = distinct_leads_by_day.get(day, 0)
        incoming_counts = incoming_by_day.get(day, {})
        outgoing_counts = outgoing_by_day.get(day, {})
        outcome_table.append(
            {
                "date": day,
                "totalDurationSec": total_duration,
                "avgDurationPerLeadSec": round(total_duration / leads, 1) if leads else None,
                "incoming": {b: incoming_counts.get(b, 0) for b in _INCOMING_BUCKETS},
                "outgoing": {b: outgoing_counts.get(b, 0) for b in _OUTGOING_BUCKETS},
            }
        )

    duration_by_owner = {row["_id"]: row["totalDuration"] for row in result["durationByOwner"]}

    incoming_by_owner: dict[int, dict[str, int]] = {}
    for row in result["incomingOutcomeByOwner"]:
        incoming_by_owner.setdefault(row["_id"]["ownerId"], {})[row["_id"]["bucket"]] = row["count"]

    outgoing_by_owner: dict[int, dict[str, int]] = {}
    for row in result["outgoingOutcomeByOwner"]:
        outgoing_by_owner.setdefault(row["_id"]["ownerId"], {})[row["_id"]["bucket"]] = row["count"]

    # Same shape as outcome_table, keyed by owner instead of day — feeds
    # the telecaller-wise Outcome tab.
    outcome_table_by_owner = []
    for owner_id in owner_ids:
        total_duration = duration_by_owner.get(owner_id, 0)
        leads = distinct_leads_by_owner_map.get(owner_id, 0)
        incoming_counts = incoming_by_owner.get(owner_id, {})
        outgoing_counts = outgoing_by_owner.get(owner_id, {})
        outcome_table_by_owner.append(
            {
                "ownerId": owner_id,
                "totalDurationSec": total_duration,
                "avgDurationPerLeadSec": round(total_duration / leads, 1) if leads else None,
                "incoming": {b: incoming_counts.get(b, 0) for b in _INCOMING_BUCKETS},
                "outgoing": {b: outgoing_counts.get(b, 0) for b in _OUTGOING_BUCKETS},
            }
        )

    return {
        "days": days,
        "outcomeByDay": outcome_rows,
        "callsPerLeadByDay": calls_per_lead_rows,
        "outcomeByOwner": outcome_by_owner_rows,
        "callsPerLeadByOwner": calls_per_lead_by_owner_rows,
        "leadAgeByDay": lead_age_rows,
        "distinctLeads": distinct_leads,
        "distinctLeadsByOwner": distinct_leads_by_owner_rows,
        "outcomeTable": outcome_table,
        "outcomeTableByOwner": outcome_table_by_owner,
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
