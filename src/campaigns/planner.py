
from datetime import datetime, timezone
from src.tracking.mlflow_utils import log_prediction

def create_campaign_plan(hotspots, top_n=10):
    """Create a cleanup priority list from observed hotspot records."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    ranked = sorted(
        hotspots,
        key=lambda item: item["priority_score"],
        reverse=True,
    )

    campaigns = []

    for rank, hotspot in enumerate(ranked[:top_n], start=1):
        campaigns.append({
            "rank": rank,
            "site_id": hotspot["site_id"],
            "latitude": hotspot["latitude"],
            "longitude": hotspot["longitude"],
            "priority_score": hotspot["priority_score"],
            "total_detections": hotspot["total_detections"],
            "waste_types": hotspot["waste_types"],
            "recommended_action": (
                "High-priority cleanup inspection"
                if rank <= 3
                else "Schedule cleanup inspection"
            ),
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_hotspots": len(hotspots),
        "campaigns": campaigns,
    }
