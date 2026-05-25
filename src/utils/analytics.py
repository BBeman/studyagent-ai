"""
Usage Analytics Logger - writes every interaction to DynamoDB.

Captures: timestamp, session_id, module, question preview, tools used,
response time, verification scores, and optional thumbs up/down feedback.

Table: studyagent-analytics (PK: interaction_id)
"""
import logging
import uuid
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal

from src.config import AWS_REGION, ANALYTICS_TABLE_NAME, get_boto3_session

logger = logging.getLogger("analytics")

_table = None


def _get_table():
    global _table
    if _table is None:
        session = get_boto3_session()
        dynamodb = session.resource("dynamodb", region_name=AWS_REGION)
        _table = dynamodb.Table(ANALYTICS_TABLE_NAME)
    return _table


def log_interaction(
    session_id: str,
    module: str,
    question: str,
    tools_used: list,
    response_time_ms: int,
    verification_grounded: str = "N/A",
    verification_confidence: str = "low",
) -> str:
    """Log a chat interaction to DynamoDB. Returns the interaction_id."""
    interaction_id = str(uuid.uuid4())
    item = {
        "interaction_id": interaction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "module": module or "General",
        "question_preview": question[:100],
        "tools_used": tools_used if tools_used else [],
        "response_time_ms": Decimal(str(response_time_ms)),
        "verification_grounded": verification_grounded,
        "verification_confidence": verification_confidence,
    }
    try:
        _get_table().put_item(Item=item)
        logger.info(f"Logged interaction {interaction_id}")
    except Exception as e:
        logger.warning(f"Failed to log analytics: {e}")
    return interaction_id


def record_feedback(interaction_id: str, feedback: str) -> bool:
    """Record thumbs up/down feedback on an existing interaction."""
    try:
        _get_table().update_item(
            Key={"interaction_id": interaction_id},
            UpdateExpression="SET feedback = :f, feedback_ts = :t",
            ExpressionAttributeValues={
                ":f": feedback,
                ":t": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info(f"Recorded feedback '{feedback}' for {interaction_id}")
        return True
    except Exception as e:
        logger.warning(f"Failed to record feedback: {e}")
        return False


def get_analytics_summary() -> dict:
    """Scan the analytics table and return aggregate pilot metrics."""
    try:
        table = _get_table()
        items = []
        response = table.scan()
        items.extend(response.get("Items", []))
        while "LastEvaluatedKey" in response:
            response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            items.extend(response.get("Items", []))

        if not items:
            return {"total_interactions": 0}

        # Unique sessions & modules
        sessions = set()
        modules = Counter()
        tools = Counter()
        grounded = Counter()
        feedback_counts = Counter()
        response_times = []

        for item in items:
            sessions.add(item.get("session_id", ""))
            modules[item.get("module", "General")] += 1

            for t in item.get("tools_used", []):
                tools[t] += 1

            g = item.get("verification_grounded", "N/A").lower()
            grounded[g] += 1

            fb = item.get("feedback")
            if fb:
                feedback_counts[fb] += 1

            rt = item.get("response_time_ms")
            if rt is not None:
                response_times.append(float(rt))

        total = len(items)
        verified_total = grounded.get("yes", 0) + grounded.get("partial", 0) + grounded.get("no", 0)
        verification_score = (
            round((grounded.get("yes", 0) + grounded.get("partial", 0)) / verified_total * 100, 1)
            if verified_total > 0
            else None
        )

        return {
            "total_interactions": total,
            "unique_sessions": len(sessions),
            "modules": dict(modules.most_common()),
            "tools_used": dict(tools.most_common()),
            "verification": {
                "score_pct": verification_score,
                "breakdown": dict(grounded),
            },
            "feedback": {
                "thumbs_up": feedback_counts.get("up", 0),
                "thumbs_down": feedback_counts.get("down", 0),
                "no_feedback": total - sum(feedback_counts.values()),
            },
            "response_time_ms": {
                "avg": round(sum(response_times) / len(response_times)) if response_times else 0,
                "min": round(min(response_times)) if response_times else 0,
                "max": round(max(response_times)) if response_times else 0,
            },
        }
    except Exception as e:
        logger.warning(f"Failed to get analytics summary: {e}")
        return {"error": str(e)}
