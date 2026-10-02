"""FastAPI backend for the CRM call-coverage dashboard.

Reads directly from the `kylas` MongoDB Atlas database via the async
`motor` driver. No LLM calls anywhere in the data path.
"""
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import List, Optional

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
    # Backs the $match in /api/lead-dashboard, which filters on any
    # combination of these four fields — a range on createdAt leads the
    # compound index since it's the filter every dashboard load applies.
    await app.state.db.leads.create_index(
        [("createdAt", 1), ("pipeline", 1), ("utmSource", 1), ("utmMedium", 1)]
    )
    # Backs the {"date": ...} lookup in /api/google-ad-funnel.
    await app.state.db.campaign_data.create_index("date")
    # Backs the manager-scoped utmSource $in match behind the funnel's filter
    # options — without it every dropdown load scans the whole leads collection.
    await app.state.db.leads.create_index("utmSource")
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


# Small in-process TTL cache for read-heavy endpoints whose inputs change
# rarely (filter lists, uploaded-date list) or are re-requested constantly
# (the same funnel view across page loads). Single-process only — fine for
# this deployment; swap for Redis if the API ever runs multi-worker.
_CACHE_MAX_ENTRIES = 500
_cache: dict = {}


async def _cached(key, ttl_sec: float, loader):
    now = time.monotonic()
    hit = _cache.get(key)
    if hit is not None and now < hit[0]:
        return hit[1]
    value = await loader()
    if len(_cache) >= _CACHE_MAX_ENTRIES:
        for k in [k for k, (exp, _) in _cache.items() if exp <= now]:
            del _cache[k]
        if len(_cache) >= _CACHE_MAX_ENTRIES:
            _cache.clear()
    _cache[key] = (now + ttl_sec, value)
    return value


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


# Sent by the frontend in place of a real value to mean "leads where this
# field was never set" — a real analytics segment (e.g. ~80% of leads have
# no UTM source), not noise to hide, so it's a selectable option rather than
# something the filter list silently drops.
UNSET_FILTER = "__unset__"


def _distinct_with_unset(values: list, blank_is_unset: bool = False) -> list:
    """Distinct-value list for a filter dropdown, keeping "not set" as an
    explicit option (surfaced as `null` in the response) instead of
    dropping it — those leads matter for analytics and need to stay
    filterable. `blank_is_unset` folds "" in with None for string fields
    where the CRM sometimes stores an empty string rather than omitting the
    field entirely; both mean the same thing to an analyst.
    """
    is_unset = (lambda v: v is None or v == "") if blank_is_unset else (lambda v: v is None)
    has_unset = any(is_unset(v) for v in values)
    real = sorted(v for v in values if not is_unset(v))
    return ([None] + real) if has_unset else real


@app.get("/api/lead-dashboard/filter-options")
async def lead_dashboard_filter_options():
    """Distinct values for the Lead Dashboard's Pipeline/UTM filters.

    Pulled straight from `leads` rather than hardcoded — the CRM has no
    separate pipeline/UTM lookup collection, so whatever values actually
    appear on a lead are the only valid filter options.
    """
    pipelines = await app.state.db.leads.distinct("pipeline")
    utm_sources = await app.state.db.leads.distinct("utmSource")
    utm_mediums = await app.state.db.leads.distinct("utmMedium")
    utm_campaigns = await app.state.db.leads.distinct("utmCampaign")
    return {
        "pipelines": _distinct_with_unset(pipelines),
        "utmSources": _distinct_with_unset(utm_sources, blank_is_unset=True),
        "utmMediums": _distinct_with_unset(utm_mediums, blank_is_unset=True),
        "utmCampaigns": _distinct_with_unset(utm_campaigns, blank_is_unset=True),
    }


def _parse_csv(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip() != ""]


def _resolve_unset_values(tokens: list[str]) -> list:
    """Expands the `UNSET_FILTER` token into the actual "not set" values it
    stands for (None, and "" for string fields that use blank rather than
    absent) so callers can drop the result straight into a Mongo `$in` —
    which already matches missing/null fields for a None in the list.
    """
    expanded = []
    for token in tokens:
        if token == UNSET_FILTER:
            expanded.extend([None, ""])
        else:
            expanded.append(token)
    return expanded


def _parse_int_csv(raw: Optional[str], field_name: str) -> list:
    """Comma-separated integer codes, plus `UNSET_FILTER` standing in for
    "field not set" (kept as None, not dropped — see UNSET_FILTER)."""
    values = []
    for part in _parse_csv(raw):
        if part == UNSET_FILTER:
            values.append(None)
            continue
        try:
            values.append(int(part))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} must be a comma-separated list of integers (or '{UNSET_FILTER}')",
            )
    return values


def _parse_single_int_field(raw: Optional[str], field_name: str, match: dict, key: str) -> None:
    """Sets `match[key]` from a single-value int query param that may also be
    `UNSET_FILTER` (see UNSET_FILTER) — left out of `match` entirely when the
    param wasn't given at all, which is different from asking to filter for
    the field being unset.
    """
    if raw is None:
        return
    if raw == UNSET_FILTER:
        match[key] = None
        return
    try:
        match[key] = int(raw)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be an integer (or '{UNSET_FILTER}')",
        )


@app.get("/api/lead-dashboard")
async def lead_dashboard(
    start: Optional[date] = Query(None, description="Lead createdAt lower bound"),
    end: Optional[date] = Query(None, description="Lead createdAt upper bound"),
    pipeline: Optional[str] = Query(
        None, description=f"Pipeline id, or '{UNSET_FILTER}' for leads with none set"
    ),
    utmSource: Optional[str] = Query(None, description="Comma-separated list"),
    utmMedium: Optional[str] = Query(None, description="Comma-separated list"),
    utmCampaign: Optional[str] = Query(None, description="Comma-separated list"),
    ownerIds: Optional[str] = Query(None, description="Comma-separated list of CRM owner ids"),
    otpStatus: Optional[str] = Query(None),
    registrationStatus: Optional[str] = Query(None, description="'registered' or 'unregistered'"),
    examMode: Optional[str] = Query(
        None, description=f"Exam mode code, or '{UNSET_FILTER}' for leads with none set"
    ),
    course: Optional[str] = Query(None, description="Comma-separated list of course codes"),
    class_: Optional[str] = Query(None, alias="class", description="Comma-separated list of class codes"),
):
    """One aggregation, faceted, backing every widget on the Lead Dashboard.

    Every filter is optional and AND together — the same filtered document
    set then feeds every facet below, so summary cards, OTP/exam-mode/
    course/class distributions, and the registration split can never
    disagree with each other or cost a second round trip.

    Date/pipeline/utmSource/utmMedium/utmCampaign/ownerIds run in the first
    $match so date/pipeline/utmSource/utmMedium can use the
    leads.createdAt/pipeline/utmSource/utmMedium compound index (utmCampaign
    and ownerId aren't part of that index but are cheap raw-field matches,
    same as the rest of this stage).
    otpStatus/registrationStatus filter on fields this endpoint derives
    itself (_cfOtpStatus/_isRegistered — see the $addFields below, notably
    the "Unknown" OTP fallback and the registered-if-cfRegNo-is-set rule),
    so they run in a second $match after those fields exist rather than
    duplicating that derivation logic here.

    Code fields (cfExamMode/cfCourse/cfClass) are grouped by their raw
    numeric code here; converting a code to its human label is left to the
    frontend's mapping JSON, so this endpoint has no display concerns and
    doesn't need to change if that mapping does.
    """
    if start is not None and end is not None and end < start:
        raise HTTPException(status_code=400, detail="end must be on or after start")
    if registrationStatus is not None and registrationStatus not in ("registered", "unregistered"):
        raise HTTPException(
            status_code=400, detail="registrationStatus must be 'registered' or 'unregistered'"
        )

    utm_sources = _resolve_unset_values(_parse_csv(utmSource))
    utm_mediums = _resolve_unset_values(_parse_csv(utmMedium))
    utm_campaigns = _resolve_unset_values(_parse_csv(utmCampaign))
    owner_ids = _parse_int_csv(ownerIds, "ownerIds")
    course_codes = _parse_int_csv(course, "course")
    class_codes = _parse_int_csv(class_, "class")

    match: dict = {}
    if start is not None or end is not None:
        created_at: dict = {}
        if start is not None:
            created_at["$gte"] = f"{start.isoformat()}T00:00:00.000Z"
        if end is not None:
            created_at["$lte"] = f"{end.isoformat()}T23:59:59.999Z"
        match["createdAt"] = created_at
    _parse_single_int_field(pipeline, "pipeline", match, "pipeline")
    if utm_sources:
        match["utmSource"] = {"$in": utm_sources}
    # utmMedium is applied inside each facet except its own (see below), so
    # the medium donut keeps showing every medium — with the selected ones
    # highlighted — instead of collapsing to just the clicked slice.
    if utm_campaigns:
        match["utmCampaign"] = {"$in": utm_campaigns}
    if owner_ids:
        match["ownerId"] = {"$in": owner_ids}

    pipeline_stages = [
        {"$match": match},
        {
            "$addFields": {
                # A lead with no customFieldValues at all (never touched the
                # registration form) hits the same $ifNull fallback as one
                # whose cfRegNo key is present but explicitly null.
                "_cfRegNo": {"$ifNull": ["$customFieldValues.cfRegNo", None]},
                "_cfOtpStatus": {
                    "$ifNull": ["$customFieldValues.cfOtpStatus", "Unknown"]
                },
            }
        },
        {
            "$addFields": {
                "_isRegistered": {
                    "$and": [
                        {"$ne": ["$_cfRegNo", None]},
                        {"$ne": ["$_cfRegNo", ""]},
                    ]
                },
                "_isOtpVerified": {"$eq": ["$_cfOtpStatus", "Verified"]},
            }
        },
    ]

    post_match: dict = {}
    if otpStatus:
        post_match["_cfOtpStatus"] = otpStatus
    if registrationStatus:
        post_match["_isRegistered"] = registrationStatus == "registered"
    _parse_single_int_field(examMode, "examMode", post_match, "customFieldValues.cfExamMode")
    if course_codes:
        post_match["customFieldValues.cfCourse"] = {"$in": course_codes}
    if class_codes:
        post_match["customFieldValues.cfClass"] = {"$in": class_codes}
    if post_match:
        pipeline_stages.append({"$match": post_match})

    medium_stage = [{"$match": {"utmMedium": {"$in": utm_mediums}}}] if utm_mediums else []

    pipeline_stages.append(
        {
            "$facet": {
                "totals": medium_stage + [
                    {
                        "$group": {
                            "_id": None,
                            "totalLeads": {"$sum": 1},
                            "registeredLeads": {
                                "$sum": {"$cond": ["$_isRegistered", 1, 0]}
                            },
                            "otpVerifiedLeads": {
                                "$sum": {"$cond": ["$_isOtpVerified", 1, 0]}
                            },
                        }
                    }
                ],
                "otpStatus": medium_stage + [
                    {"$group": {"_id": "$_cfOtpStatus", "count": {"$sum": 1}}}
                ],
                "examMode": medium_stage + [
                    {
                        "$group": {
                            "_id": {"$ifNull": ["$customFieldValues.cfExamMode", None]},
                            "count": {"$sum": 1},
                        }
                    }
                ],
                "utmMedium": [
                    {"$group": {"_id": {"$ifNull": ["$utmMedium", None]}, "count": {"$sum": 1}}}
                ],
                "course": medium_stage + [
                    {
                        "$group": {
                            "_id": {"$ifNull": ["$customFieldValues.cfCourse", None]},
                            "count": {"$sum": 1},
                        }
                    }
                ],
                "class": medium_stage + [
                    {
                        "$group": {
                            "_id": {"$ifNull": ["$customFieldValues.cfClass", None]},
                            "count": {"$sum": 1},
                        }
                    }
                ],
            }
        },
    )

    [result] = await app.state.db.leads.aggregate(pipeline_stages).to_list(length=1)

    totals = result["totals"][0] if result["totals"] else None
    total_leads = totals["totalLeads"] if totals else 0
    registered_leads = totals["registeredLeads"] if totals else 0
    otp_verified_leads = totals["otpVerifiedLeads"] if totals else 0

    def dist(rows: list) -> list[dict]:
        # code is None for leads with no value on that custom field — the
        # frontend renders that bucket as "Unspecified" rather than
        # dropping those leads from the chart.
        return [{"code": row["_id"], "count": row["count"]} for row in rows]

    return {
        "totalLeads": total_leads,
        "registeredLeads": registered_leads,
        "unregisteredLeads": total_leads - registered_leads,
        "otpVerifiedLeads": otp_verified_leads,
        "otpStatusDistribution": dist(result["otpStatus"]),
        "examModeDistribution": dist(result["examMode"]),
        "utmMediumDistribution": dist(result["utmMedium"]),
        "courseDistribution": dist(result["course"]),
        "classDistribution": dist(result["class"]),
    }


# ---------------------------------------------------------------------------
# Campaign summary
# ---------------------------------------------------------------------------

# Campaign Summary is scoped to utmSource exactly "g" (not the longer
# "google" spelling the Google Ad Funnel also accepts), and splits those leads
# by utmMedium into the four Google campaign types. A lead with a blank or
# unrecognised medium lands in "other" so the totals still reconcile with the
# source's full lead count; that row is only shown when it's non-empty.
_CAMPAIGN_SUMMARY_SOURCE = "g"
_CAMPAIGN_SUMMARY_MEDIUMS = {
    "dis": "display",
    "pmax": "pmax",
    "yt": "demand_gen",
    "sea": "search",
}
_CAMPAIGN_SUMMARY_LABELS = {
    "demand_gen": "Demand Gen",
    "display": "Display",
    "pmax": "Performance Max",
    "search": "Search",
    "other": "Other / blank",
}
_CAMPAIGN_SUMMARY_ORDER = list(_CAMPAIGN_SUMMARY_LABELS)


@app.get("/api/campaign-summary")
async def campaign_summary(
    start: date = Query(..., description="Lead createdAt lower bound"),
    end: date = Query(..., description="Lead createdAt upper bound"),
):
    """Per-day, per-campaign-type lead counts for utmSource "g".

    One row per (date, category): inquiries (every lead), registered
    (cfRegNo set — same rule as /api/lead-dashboard) and not registered.
    Every date in the range gets a row for every category, zeros included,
    so N dates x M categories is always exactly N*M rows.
    """
    if end < start:
        raise HTTPException(status_code=400, detail="end must be on or after start")

    async def load():
        pipeline = [
            {
                "$match": {
                    "utmSource": _CAMPAIGN_SUMMARY_SOURCE,
                    "createdAt": {
                        "$gte": f"{start.isoformat()}T00:00:00.000Z",
                        "$lte": f"{end.isoformat()}T23:59:59.999Z",
                    },
                }
            },
            {
                "$group": {
                    "_id": {
                        # createdAt is an ISO-8601 string; its first 10
                        # characters are the calendar date (same as the
                        # other date-wise endpoints).
                        "day": {"$substrCP": ["$createdAt", 0, 10]},
                        "medium": {"$ifNull": ["$utmMedium", ""]},
                    },
                    "inquiries": {"$sum": 1},
                    "registered": {
                        "$sum": {
                            "$cond": [
                                {"$ne": [{"$ifNull": ["$customFieldValues.cfRegNo", ""]}, ""]},
                                1,
                                0,
                            ]
                        }
                    },
                }
            },
        ]
        grouped = await app.state.db.leads.aggregate(pipeline).to_list(length=None)

        cells: dict[tuple[str, str], dict[str, int]] = {}
        for row in grouped:
            category = _CAMPAIGN_SUMMARY_MEDIUMS.get(row["_id"]["medium"].strip().lower(), "other")
            cell = cells.setdefault((row["_id"]["day"], category), {"inquiries": 0, "registered": 0})
            cell["inquiries"] += row["inquiries"]
            cell["registered"] += row["registered"]

        has_other = any(category == "other" for _, category in cells)
        categories = [c for c in _CAMPAIGN_SUMMARY_ORDER if c != "other" or has_other]

        rows = []
        for day in _date_range(start, end):
            for category in categories:
                cell = cells.get((day, category), {"inquiries": 0, "registered": 0})
                rows.append(
                    {
                        "date": day,
                        "category": category,
                        "label": _CAMPAIGN_SUMMARY_LABELS[category],
                        "inquiries": cell["inquiries"],
                        "registered": cell["registered"],
                        "notRegistered": cell["inquiries"] - cell["registered"],
                    }
                )

        inquiries = sum(r["inquiries"] for r in rows)
        registered = sum(r["registered"] for r in rows)
        return {
            "utmSource": _CAMPAIGN_SUMMARY_SOURCE,
            "rows": rows,
            "total": {
                "inquiries": inquiries,
                "registered": registered,
                "notRegistered": inquiries - registered,
            },
        }

    return await _cached(("campaign-summary", start, end), _FUNNEL_TTL_SEC, load)


# ---------------------------------------------------------------------------
# Google Ads funnel
# ---------------------------------------------------------------------------

# Leads carry a free-text utmCampaign (ME, b_me, nb_f, ...) that never matches
# the Google Ads campaign names in `campaign_data`, so a lead can't be joined
# to one campaign. They CAN be placed in a coarser "segment" (campaign type,
# with Search split into branded / non-branded) from utmMedium + utmCampaign,
# and each campaign is placed in the same segment from its type + name. Lead
# metrics are therefore reported per segment; ad metrics per campaign.
_SEGMENT_LABELS = {
    "display": "Display",
    "pmax": "Performance Max",
    "search_branded": "Search · Branded",
    "search_nonbranded": "Search · Non-branded",
    "search_other": "Search · Other",
    "demand_gen": "Demand Gen / YouTube",
    "other": "Other campaigns",
    "unattributed": "Unattributed",
}
_SEGMENT_ORDER = list(_SEGMENT_LABELS)

_GOOGLE_SOURCE_REGEX = "^(g|google)$"

# `campaign_data` rows belong to a manager, and each manager's campaigns are
# tagged on leads by a utmSource. That tag isn't stored on campaign_data, so
# the manager -> utmSource link lives here: selecting a manager scopes the
# leads to these sources. A manager with no entry has no attributable leads
# (an empty list matches nothing) rather than silently inheriting everyone's.
_MANAGER_UTM_SOURCES: dict[str, list[str]] = {
    "Ayesha": ["g"],
}


def _lead_source_clause(managers: list[str]) -> dict:
    """utmSource clause for the funnel's leads: the selected managers'
    sources, or — with no manager selected — any Google-sourced lead."""
    if not managers:
        return {"utmSource": {"$regex": _GOOGLE_SOURCE_REGEX, "$options": "i"}}
    sources = sorted({src for m in managers for src in _MANAGER_UTM_SOURCES.get(m, [])})
    return {"utmSource": {"$in": sources}}
_BRANDED_UTM_CAMPAIGNS = {"b", "b_me"}
_NONBRANDED_UTM_CAMPAIGNS = {"nb_f"}


def _campaign_segment(campaign_type: str, name: str) -> str:
    kind = (campaign_type or "").upper()
    if kind == "DISPLAY":
        return "display"
    if kind == "PERFORMANCE_MAX":
        return "pmax"
    if kind == "DEMAND_GEN":
        return "demand_gen"
    if kind == "SEARCH":
        flat = "".join(ch for ch in (name or "").lower() if ch.isalnum())
        if "nonbrand" in flat:
            return "search_nonbranded"
        if "brand" in flat:
            return "search_branded"
        return "search_other"
    return "other"


def _lead_segment(medium: Optional[str], utm_campaign: Optional[str]) -> str:
    m = (medium or "").strip().lower()
    c = (utm_campaign or "").strip().lower()
    if m in ("dis", "display"):
        return "display"
    if m in ("pmax", "performance_max") or (m == "cpc" and "pmax" in c):
        return "pmax"
    if m in ("yt", "youtube", "paid_video", "demandgen", "demand_gen"):
        return "demand_gen"
    if m in ("sea", "search", "cpc"):
        if c in _BRANDED_UTM_CAMPAIGNS:
            return "search_branded"
        if c in _NONBRANDED_UTM_CAMPAIGNS:
            return "search_nonbranded"
        return "search_other"
    return "unattributed"


def _num(value) -> float:
    # The Ads export stores NaN for undefined ratios (CTR, Conv. rate, ...);
    # NaN isn't valid JSON, and we recompute every ratio from sums anyway.
    if isinstance(value, (int, float)) and value == value:
        return float(value)
    return 0.0


def _ratio(numerator: float, denominator: float, digits: int = 2) -> Optional[float]:
    return round(numerator / denominator, digits) if denominator else None


def _pct(numerator: float, denominator: float) -> Optional[float]:
    return round(numerator / denominator * 100, 2) if denominator else None


def _ad_metrics(spend: float, impressions: float, clicks: float, conversions: float) -> dict:
    return {
        "spend": round(spend, 2),
        "impressions": int(impressions),
        "clicks": int(clicks),
        "conversions": round(conversions, 2),
        "ctrPct": _pct(clicks, impressions),
        "cpc": _ratio(spend, clicks),
        "convRatePct": _pct(conversions, clicks),
        "costPerConversion": _ratio(spend, conversions),
    }


def _lead_metrics(
    spend: float, leads: int, registered: int, registered_verified: int, verified: int
) -> dict:
    """`verified` = every OTP-verified lead; `registered_verified` = the
    intersection (registered AND OTP-verified), the funnel's last step."""
    # No spend means no ad data for that day/segment — a cost of 0 would read
    # as "free leads", so cost-per-X is undefined (null) rather than 0.
    if not spend:
        cpl = cost_reg = cost_ver = cost_reg_ver = None
    else:
        cpl = _ratio(spend, leads)
        cost_reg = _ratio(spend, registered)
        cost_ver = _ratio(spend, verified)
        cost_reg_ver = _ratio(spend, registered_verified)
    return {
        "leads": leads,
        "registeredLeads": registered,
        "unregisteredLeads": leads - registered,
        "verifiedLeads": verified,
        "unverifiedLeads": leads - verified,
        "registeredVerifiedLeads": registered_verified,
        "cpl": cpl,
        "costPerRegisteredLead": cost_reg,
        "costPerVerifiedLead": cost_ver,
        "costPerRegisteredVerifiedLead": cost_reg_ver,
        "registrationRatePct": _pct(registered, leads),
        # Of the registered leads, how many are verified — the step the funnel shows.
        "verificationRatePct": _pct(registered_verified, registered),
        # Of the verified leads, how many are also registered.
        "registeredOfVerifiedPct": _pct(registered_verified, verified),
    }



_FUNNEL_TTL_SEC = 60
_FILTER_OPTIONS_TTL_SEC = 600
_FUNNEL_DATES_TTL_SEC = 300


@app.get("/api/google-ad-funnel")
async def google_ad_funnel(
    date_: Optional[date] = Query(
        None,
        alias="date",
        description="Optional single day. Omitted = everything in campaign_data "
        "(a cumulative export) and the manager's leads over all time.",
    ),
    manager: Optional[str] = Query(
        None, description="Comma-separated campaign managers; scopes campaigns AND leads"
    ),
    pipeline: Optional[str] = Query(
        None, description=f"Pipeline id, or '{UNSET_FILTER}' for leads with none set"
    ),
    utmSource: Optional[str] = Query(None, description="Comma-separated list"),
    utmMedium: Optional[str] = Query(None, description="Comma-separated list"),
    utmCampaign: Optional[str] = Query(None, description="Comma-separated list"),
    ownerIds: Optional[str] = Query(None, description="Comma-separated list of CRM owner ids"),
    otpStatus: Optional[str] = Query(None),
    registrationStatus: Optional[str] = Query(None, description="'registered' or 'unregistered'"),
    examMode: Optional[str] = Query(None),
    course: Optional[str] = Query(None, description="Comma-separated list of course codes"),
    class_: Optional[str] = Query(None, alias="class", description="Comma-separated list of class codes"),
    segment: Optional[List[str]] = Query(
        None, description="Repeatable. Segment keys to scope to (see _SEGMENT_LABELS)"
    ),
    campaign: Optional[List[str]] = Query(
        None, description="Repeatable. Campaign names to scope ad metrics to"
    ),
):
    # Cached briefly per exact parameter set: the same view is re-requested
    # on every reload/toggle, and leads only change as they're created.
    # Errors (e.g. a bad registrationStatus) raise inside the loader and are
    # never cached.
    args = (
        date_, manager, pipeline, utmSource, utmMedium, utmCampaign,
        ownerIds, otpStatus, registrationStatus, examMode, course, class_,
        tuple(sorted(segment or [])), tuple(sorted(campaign or [])),
    )
    return await _cached(
        ("funnel", *args), _FUNNEL_TTL_SEC, lambda: _compute_google_ad_funnel(*args)
    )


async def _compute_google_ad_funnel(
    date_: Optional[date],
    manager: Optional[str],
    pipeline: Optional[str],
    utmSource: Optional[str],
    utmMedium: Optional[str],
    utmCampaign: Optional[str],
    ownerIds: Optional[str],
    otpStatus: Optional[str],
    registrationStatus: Optional[str],
    examMode: Optional[str],
    course: Optional[str],
    class_: Optional[str],
    selected_segments: tuple,
    selected_campaigns: tuple,
):
    """Daily Google Ads performance + lead funnel, aggregated over campaigns.

    Ad metrics come from `campaign_data` (one document per campaign per
    `date`, when one is asked for); leads come from `leads` with the manager's utmSource,
    created that day when a date is given, otherwise over all time.
    The optional filters (same names/semantics as /api/lead-dashboard) narrow
    the LEADS only — ad spend can't be split by lead attributes — so with a
    filter active, cost-per-X is the day's full spend over the filtered leads.
    `manager` scopes both sides: that manager's campaigns, and the leads whose
    utmSource is mapped to that manager (see _MANAGER_UTM_SOURCES).
    Spend is the numerator of every cost metric, so cost-per-X is total
    spend / total X — never an average of per-campaign figures.
    """
    if registrationStatus is not None and registrationStatus not in ("registered", "unregistered"):
        raise HTTPException(
            status_code=400, detail="registrationStatus must be 'registered' or 'unregistered'"
        )
    for key in selected_segments:
        if key not in _SEGMENT_LABELS:
            raise HTTPException(status_code=400, detail=f"unknown segment '{key}'")
    day = date_.isoformat() if date_ else None
    db_ = app.state.db

    campaign_query: dict = {"date": day} if day else {}
    managers = _parse_csv(manager)
    if managers:
        campaign_query["Manager"] = {
            "$in": [None, ""] + [m for m in managers if m != UNSET_FILTER]
            if UNSET_FILTER in managers
            else managers
        }
    campaign_docs = await db_.campaign_data.find(campaign_query).to_list(length=None)

    match: dict = {
        # Always scoped to the manager's (or, with none, Google) sources,
        # whatever the utmSource filter says.
        "$and": [_lead_source_clause([m for m in managers if m != UNSET_FILTER])],
    }
    # Leads are counted over the same time window the ad data covers, so
    # cost-per-lead divides spend by leads from the same period. A single
    # `date` (if asked for) wins; otherwise the window is the span of the
    # campaign rows' start_date/end_date (earliest start .. latest end).
    starts = [d["start_date"] for d in campaign_docs if d.get("start_date")]
    ends = [d["end_date"] for d in campaign_docs if d.get("end_date")]
    window_start = day or (min(starts) if starts else None)
    window_end = day or (max(ends) if ends else None)
    created_at: dict = {}
    if window_start:
        created_at["$gte"] = f"{window_start}T00:00:00.000Z"
    if window_end:
        created_at["$lte"] = f"{window_end}T23:59:59.999Z"
    if created_at:
        match["createdAt"] = created_at
    utm_sources = _resolve_unset_values(_parse_csv(utmSource))
    utm_mediums = _resolve_unset_values(_parse_csv(utmMedium))
    utm_campaigns = _resolve_unset_values(_parse_csv(utmCampaign))
    owner_ids = _parse_int_csv(ownerIds, "ownerIds")
    course_codes = _parse_int_csv(course, "course")
    class_codes = _parse_int_csv(class_, "class")
    if utm_sources:
        match["$and"].append({"utmSource": {"$in": utm_sources}})
    if utm_mediums:
        match["utmMedium"] = {"$in": utm_mediums}
    if utm_campaigns:
        match["utmCampaign"] = {"$in": utm_campaigns}
    if owner_ids:
        match["ownerId"] = {"$in": owner_ids}
    _parse_single_int_field(pipeline, "pipeline", match, "pipeline")

    stages: list = [
        {"$match": match},
        {
            "$addFields": {
                "_cfOtpStatus": {"$ifNull": ["$customFieldValues.cfOtpStatus", "Unknown"]},
                # Same definition as /api/lead-dashboard: registered once
                # cfRegNo is set.
                "_isRegistered": {
                    "$ne": [{"$ifNull": ["$customFieldValues.cfRegNo", ""]}, ""]
                },
            }
        },
    ]
    post_match: dict = {}
    if otpStatus:
        post_match["_cfOtpStatus"] = otpStatus
    if registrationStatus:
        post_match["_isRegistered"] = registrationStatus == "registered"
    _parse_single_int_field(examMode, "examMode", post_match, "customFieldValues.cfExamMode")
    if course_codes:
        post_match["customFieldValues.cfCourse"] = {"$in": course_codes}
    if class_codes:
        post_match["customFieldValues.cfClass"] = {"$in": class_codes}
    if post_match:
        stages.append({"$match": post_match})
    base_stages = list(stages)  # match + derived fields + post-match, before grouping
    stages.append(
        {
            "$group": {
                "_id": {"medium": "$utmMedium", "campaign": "$utmCampaign"},
                "leads": {"$sum": 1},
                "registered": {"$sum": {"$cond": ["$_isRegistered", 1, 0]}},
                # `verified` is the INTERSECTION (registered AND OTP-verified) — the
                # funnel's last step, so it always narrows (Leads >= Registered >=
                # this). `verifiedAny` is every OTP-verified lead, registered or not.
                "verified": {
                    "$sum": {
                        "$cond": [
                            {"$and": ["$_isRegistered", {"$eq": ["$_cfOtpStatus", "Verified"]}]},
                            1,
                            0,
                        ]
                    }
                },
                "verifiedAny": {
                    "$sum": {"$cond": [{"$eq": ["$_cfOtpStatus", "Verified"]}, 1, 0]}
                },
            }
        }
    )
    lead_rows = await db_.leads.aggregate(stages).to_list(length=None)

    segments: dict[str, dict] = {}

    def segment(key: str) -> dict:
        return segments.setdefault(
            key,
            {
                "key": key,
                "label": _SEGMENT_LABELS[key],
                "spend": 0.0,
                "impressions": 0.0,
                "clicks": 0.0,
                "conversions": 0.0,
                "leads": 0,
                "registered": 0,
                "verified": 0,
                "verified_any": 0,
                "campaigns": [],
            },
        )

    budget_by_campaign: dict[str, float] = {}
    enabled_campaigns = 0
    for doc in campaign_docs:
        name = doc.get("Campaign") or "(unnamed)"
        spend = _num(doc.get("Cost"))
        impressions = _num(doc.get("Impr."))
        clicks = _num(doc.get("Clicks"))
        conversions = _num(doc.get("Conversions"))
        budget = _num(doc.get("Budget"))
        status = doc.get("Campaign status") or "UNKNOWN"
        if status == "ENABLED":
            enabled_campaigns += 1
        # Budget is taken per campaign (the `Campaign` column), not per
        # `Budget name`: campaigns on a shared portfolio budget each count
        # their own Budget value.
        budget_by_campaign[name] = budget

        seg = segment(_campaign_segment(doc.get("Campaign type"), name))
        seg["spend"] += spend
        seg["impressions"] += impressions
        seg["clicks"] += clicks
        seg["conversions"] += conversions
        seg["campaigns"].append(
            {
                "name": name,
                "status": status,
                "type": doc.get("Campaign type"),
                "manager": doc.get("Manager"),
                "budget": round(budget, 2),
                "convValue": round(_num(doc.get("Conv. value")), 2),
                **_ad_metrics(spend, impressions, clicks, conversions),
            }
        )

    total_verified_any = 0
    for row in lead_rows:
        total_verified_any += row["verifiedAny"]
        seg = segment(_lead_segment(row["_id"].get("medium"), row["_id"].get("campaign")))
        seg["leads"] += row["leads"]
        seg["registered"] += row["registered"]
        seg["verified"] += row["verified"]
        seg["verified_any"] += row["verifiedAny"]

    total_spend = sum(s["spend"] for s in segments.values())
    total_impr = sum(s["impressions"] for s in segments.values())
    total_clicks = sum(s["clicks"] for s in segments.values())
    total_conv = sum(s["conversions"] for s in segments.values())
    total_leads = sum(s["leads"] for s in segments.values())
    total_registered = sum(s["registered"] for s in segments.values())
    total_verified = sum(s["verified"] for s in segments.values())
    total_budget = sum(budget_by_campaign.values())

    segment_rows = []
    for key in _SEGMENT_ORDER:
        seg = segments.get(key)
        if seg is None:
            continue
        seg["campaigns"].sort(key=lambda c: c["spend"], reverse=True)
        segment_rows.append(
            {
                "key": seg["key"],
                "label": seg["label"],
                **_ad_metrics(seg["spend"], seg["impressions"], seg["clicks"], seg["conversions"]),
                **_lead_metrics(
                    seg["spend"], seg["leads"], seg["registered"], seg["verified"], seg["verified_any"]
                ),
                "campaigns": seg["campaigns"],
            }
        )
    segment_rows.sort(key=lambda s: (s["spend"], s["leads"]), reverse=True)

    def build_overall(ad, leads, verified_any, budget, cost_spend):
        """KPI block. `cost_spend` is the spend the cost-per-lead metrics divide
        (0 -> they come back null): it differs from the displayed spend when a
        single campaign is selected, because its leads can't be isolated."""
        spend, impr, clicks, conv = ad
        n_leads, n_reg, n_ver = leads
        return {
            **_ad_metrics(spend, impr, clicks, conv),
            **_lead_metrics(cost_spend, n_leads, n_reg, n_ver, verified_any),
            # Sum of each campaign's DAILY budget — not comparable with spend,
            # which may be cumulative, so no utilisation figure is derived.
            "budget": round(budget, 2),
            "otpVerifiedUnregistered": verified_any - n_ver,
            # Click -> lead: how many ad clicks ended up as a CRM lead.
            "leadConversionRatePct": _pct(n_leads, clicks),
            "leadToVerifiedPct": _pct(n_ver, n_leads),
        }

    grand = build_overall(
        (total_spend, total_impr, total_clicks, total_conv),
        (total_leads, total_registered, total_verified),
        total_verified_any,
        total_budget,
        total_spend,
    )

    # ---- selection (multi): any mix of whole segments and single campaigns.
    # Ad metrics cover every selected campaign (a selected segment brings all
    # of its campaigns). Leads exist only per segment, so they cover every
    # segment that is selected or contains a selected campaign. Cost-per-lead
    # is only meaningful when each contributing segment is FULLY on the ad
    # side too — otherwise spend and leads describe different things — so it
    # is blank when a segment is only partly selected.
    camp_segment = {c["name"]: key for key, seg in segments.items() for c in seg["campaigns"]}
    sel_segments = set(selected_segments)
    sel_campaigns = {n for n in selected_campaigns if n in camp_segment}
    has_selection = bool(sel_segments or sel_campaigns)

    if has_selection:
        lead_scope = set(sel_segments) | {camp_segment[n] for n in sel_campaigns}
        scope_rows = []
        cost_available = True
        for key in lead_scope:
            seg = segments.get(key)
            if seg is None:
                continue  # selected key with no campaigns and no leads
            names = {c["name"] for c in seg["campaigns"]}
            fully = key in sel_segments or names <= sel_campaigns
            cost_available = cost_available and fully
            scope_rows.extend(
                c for c in seg["campaigns"] if key in sel_segments or c["name"] in sel_campaigns
            )
        in_scope = [segments[k] for k in lead_scope if k in segments]
        sp = sum(c["spend"] for c in scope_rows)
        overall = build_overall(
            (
                sp,
                sum(c["impressions"] for c in scope_rows),
                sum(c["clicks"] for c in scope_rows),
                sum(c["conversions"] for c in scope_rows),
            ),
            (
                sum(g["leads"] for g in in_scope),
                sum(g["registered"] for g in in_scope),
                sum(g["verified"] for g in in_scope),
            ),
            sum(g["verified_any"] for g in in_scope),
            sum(c["budget"] for c in scope_rows),
            sp if cost_available else 0,
        )
    else:
        lead_scope = set()
        cost_available = True
        overall = grand

    funnel = [
        {"stage": "Spend", "value": overall["spend"], "isCurrency": True},
        {
            "stage": "Leads",
            "value": overall["leads"],
            "stepRatePct": None,
            "costPer": overall["cpl"],
        },
        {
            "stage": "Registered",
            "value": overall["registeredLeads"],
            "stepRatePct": overall["registrationRatePct"],
            "costPer": overall["costPerRegisteredLead"],
        },
        {
            "stage": "Registered & verified",
            "value": overall["registeredVerifiedLeads"],
            "stepRatePct": overall["verificationRatePct"],
            "costPer": overall["costPerRegisteredVerifiedLead"],
        },
    ]

    # ---- class / course distribution of the leads in scope (all the lead
    # filters + the selected segment). The segment is a function of
    # (utmMedium, utmCampaign), so it's applied as the set of pairs that map to it.
    dist_stages = list(base_stages)
    run_dist = True
    if has_selection:
        pairs = [
            {"utmMedium": r["_id"].get("medium"), "utmCampaign": r["_id"].get("campaign")}
            for r in lead_rows
            if _lead_segment(r["_id"].get("medium"), r["_id"].get("campaign")) in lead_scope
        ]
        if pairs:
            dist_stages.append({"$match": {"$or": pairs}})
        else:
            run_dist = False
    distributions = {"class": [], "course": []}
    if run_dist:
        [dist] = await db_.leads.aggregate(
            dist_stages
            + [
                {
                    "$facet": {
                        "class": [
                            {"$group": {"_id": {"$ifNull": ["$customFieldValues.cfClass", None]}, "count": {"$sum": 1}}}
                        ],
                        "course": [
                            {"$group": {"_id": {"$ifNull": ["$customFieldValues.cfCourse", None]}, "count": {"$sum": 1}}}
                        ],
                    }
                }
            ]
        ).to_list(length=1)
        for key in ("class", "course"):
            distributions[key] = [{"code": r["_id"], "count": r["count"]} for r in dist[key]]

    source_clause = _lead_source_clause([m for m in managers if m != UNSET_FILTER])["utmSource"]
    return {
        "date": day,
        # What the leads were scoped to, so the page can say so. None = any
        # Google source (no manager chosen); [] = manager has no mapped source.
        "scope": {"managers": managers, "utmSources": source_clause.get("$in")},
        # The period both the ad metrics and the leads cover (YYYY-MM-DD,
        # inclusive); None when the campaign rows carry no dates.
        "window": {"startDate": window_start, "endDate": window_end},
        "selection": {
            "segments": [k for k in _SEGMENT_ORDER if k in sel_segments],
            "campaigns": sorted(sel_campaigns),
            # Labels of every segment whose leads are in view.
            "leadScope": [_SEGMENT_LABELS[k] for k in _SEGMENT_ORDER if k in lead_scope],
            # False when a segment is only partly selected (see above).
            "costAvailable": cost_available,
        },
        "hasAdData": bool(campaign_docs),
        "hasLeadData": total_leads > 0,
        # `overall`/`funnel`/`distributions` follow the selection; `segments`
        # (the table) and `tableTotal` always cover everything, so the table
        # stays a stable place to pick from.
        "overall": overall,
        "tableTotal": grand,
        "funnel": funnel,
        "distributions": distributions,
        "segments": segment_rows,
    }


@app.get("/api/google-ad-funnel/dates")
async def google_ad_funnel_dates():
    """Dates that have uploaded campaign data — lets the date picker hint at
    which days will actually show ad metrics."""
    async def load():
        return sorted(d for d in await app.state.db.campaign_data.distinct("date") if d)

    return await _cached(("funnel-dates",), _FUNNEL_DATES_TTL_SEC, load)


@app.get("/api/google-ad-funnel/filter-options")
async def google_ad_funnel_filter_options(manager: Optional[str] = Query(None)):
    """Filter-dropdown values for the funnel.

    `managers` is always returned (a cheap lookup on campaign_data). Every
    other list is distinct values among the leads in the selected manager's
    scope (see _lead_source_clause), so picking a manager first narrows every
    dropdown to options that can actually match. With no manager there is no
    lead scope yet, so those lists come back empty rather than paying for a
    full-collection scan — the page asks again once a manager is chosen.

    The lead-side lists are computed in ONE pass over the scoped leads
    ($group + $addToSet) instead of one distinct() scan per field, and cached
    per manager since they only shift as new leads/values appear.
    """
    managers_scope = _parse_csv(manager)

    async def load():
        db_ = app.state.db
        managers = _distinct_with_unset(
            await db_.campaign_data.distinct("Manager"), blank_is_unset=True
        )
        result = {
            "managers": managers,
            "utmSources": [],
            "utmMediums": [],
            "utmCampaigns": [],
            "pipelines": [],
            "ownerIds": [],
            "otpStatuses": [],
            "examModes": [],
            "courses": [],
            "classes": [],
        }
        if not managers_scope:
            return result

        [row] = await db_.leads.aggregate(
            [
                {"$match": _lead_source_clause(managers_scope)},
                {
                    "$group": {
                        "_id": None,
                        "utmSources": {"$addToSet": "$utmSource"},
                        "utmMediums": {"$addToSet": "$utmMedium"},
                        "utmCampaigns": {"$addToSet": "$utmCampaign"},
                        "pipelines": {"$addToSet": "$pipeline"},
                        "ownerIds": {"$addToSet": "$ownerId"},
                        "otpStatuses": {"$addToSet": "$customFieldValues.cfOtpStatus"},
                        "examModes": {"$addToSet": "$customFieldValues.cfExamMode"},
                        "courses": {"$addToSet": "$customFieldValues.cfCourse"},
                        "classes": {"$addToSet": "$customFieldValues.cfClass"},
                    }
                },
            ]
        ).to_list(length=1) or [None]
        if row is None:
            return result

        owner_ids = [o for o in row["ownerIds"] if o is not None]
        owners = await db_.users.find(
            {"id": {"$in": owner_ids}}, {"_id": 0, "id": 1}
        ).to_list(length=None)

        result.update(
            utmSources=_distinct_with_unset(row["utmSources"], blank_is_unset=True),
            utmMediums=_distinct_with_unset(row["utmMediums"], blank_is_unset=True),
            utmCampaigns=_distinct_with_unset(row["utmCampaigns"], blank_is_unset=True),
            pipelines=_distinct_with_unset(row["pipelines"]),
            ownerIds=sorted(o["id"] for o in owners),
            # Leads with no OTP field read as "Unknown" (same fallback as the funnel).
            otpStatuses=sorted(v for v in row["otpStatuses"] if v is not None)
            + (["Unknown"] if None in row["otpStatuses"] else []),
            # Raw codes, None = some leads have no value; the frontend maps
            # codes to labels and keeps only the ones present.
            examModes=_distinct_with_unset(row["examModes"]),
            courses=_distinct_with_unset(row["courses"]),
            classes=_distinct_with_unset(row["classes"]),
        )
        return result

    return await _cached(
        ("funnel-options", tuple(managers_scope)), _FILTER_OPTIONS_TTL_SEC, load
    )
