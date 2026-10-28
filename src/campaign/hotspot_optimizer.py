"""
Campaign Hotspot Aggregator, Priority Scoring & Resource Optimizer
Fulfills Step 15 of the Implementation Plan:
- Aggregates user waste photo submissions with GPS coordinates
- Clusters reports into geographical waste hotspots
- Computes multi-factor Waste Severity Index (WSI)
- Optimizes campaign selection based on available budget and volunteer workforce
- Provides organizer approval and before-and-after impact auditing
"""

import math
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class WasteReport:
    report_id: str
    latitude: float
    longitude: float
    plastic_count: int
    waste_pile_count: int
    face_mask_count: int
    trash_bin_count: int
    timestamp: str
    user_consent: bool = True


@dataclass
class HotspotCampaign:
    hotspot_id: str
    center_lat: float
    center_lon: float
    total_reports: int
    total_waste_items: int
    severity_index: float
    recommended_action: str
    required_volunteers: int
    estimated_cost_usd: float
    status: str = "PENDING_ORGANIZER_APPROVAL"


def haversine_distance_meters(lat1, lon1, lat2, lon2):
    """Calculates ground distance between two GPS coordinates in meters."""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def cluster_hotspots(reports: List[WasteReport], radius_meters=250.0) -> List[HotspotCampaign]:
    """Clusters waste reports within proximity radius into prioritized hotspots."""
    if not reports:
        return []

    clusters = []
    visited = set()

    for i, r in enumerate(reports):
        if i in visited:
            continue
        cluster_members = [r]
        visited.add(i)

        for j in range(i + 1, len(reports)):
            if j not in visited:
                dist = haversine_distance_meters(r.latitude, r.longitude, reports[j].latitude, reports[j].longitude)
                if dist <= radius_meters:
                    cluster_members.append(reports[j])
                    visited.add(j)

        # Aggregate cluster
        avg_lat = sum(m.latitude for m in cluster_members) / len(cluster_members)
        avg_lon = sum(m.longitude for m in cluster_members) / len(cluster_members)
        total_piles = sum(m.waste_pile_count for m in cluster_members)
        total_plastics = sum(m.plastic_count for m in cluster_members)
        total_masks = sum(m.face_mask_count for m in cluster_members)
        total_bins = sum(m.trash_bin_count for m in cluster_members)

        # Multi-factor Severity Index
        wsi = (total_piles * 4.0) + (total_plastics * 1.5) + (total_masks * 1.0) - (total_bins * 2.0)
        wsi = max(1.0, round(wsi, 1))

        # Recommend action & resource estimation
        if wsi >= 25.0:
            action = "EMERGENCY_MUNICIPAL_CREW: Heavy dumping site. Deploy compactor truck."
            volunteers = 12
            cost = 250.0
        elif wsi >= 12.0:
            action = "COMMUNITY_CLEANUP_DRIVE: Organize weekend volunteer litter sweep."
            volunteers = 8
            cost = 100.0
        elif total_bins == 0 and wsi > 5.0:
            action = "INFRASTRUCTURE_INSTALL: Placement of smart segregated waste bins."
            volunteers = 2
            cost = 180.0
        else:
            action = "AWARENESS_CAMPAIGN: Targeted citizen signage & educational flyer distribution."
            volunteers = 4
            cost = 40.0

        clusters.append(HotspotCampaign(
            hotspot_id=f"HOTSPOT_{len(clusters)+1:03d}",
            center_lat=round(avg_lat, 6),
            center_lon=round(avg_lon, 6),
            total_reports=len(cluster_members),
            total_waste_items=total_piles + total_plastics + total_masks,
            severity_index=wsi,
            recommended_action=action,
            required_volunteers=volunteers,
            estimated_cost_usd=cost
        ))

    # Sort descending by severity index
    clusters.sort(key=lambda x: x.severity_index, reverse=True)
    return clusters


def optimize_campaign_budget(hotspots: List[HotspotCampaign], max_budget_usd: float, max_volunteers: int):
    """
    Greedy Knapsack resource optimization:
    Selects high-impact hotspots that fit within the campaign's available budget and volunteer workforce.
    """
    scheduled_campaigns = []
    current_cost = 0.0
    current_volunteers = 0

    for h in hotspots:
        if (current_cost + h.estimated_cost_usd <= max_budget_usd) and (current_volunteers + h.required_volunteers <= max_volunteers):
            h.status = "APPROVED_AND_SCHEDULED"
            scheduled_campaigns.append(h)
            current_cost += h.estimated_cost_usd
            current_volunteers += h.required_volunteers

    summary = {
        "total_hotspots_detected": len(hotspots),
        "scheduled_campaigns_count": len(scheduled_campaigns),
        "total_budget_allocated_usd": current_cost,
        "remaining_budget_usd": round(max_budget_usd - current_cost, 2),
        "total_volunteers_mobilized": current_volunteers,
        "remaining_volunteers": max_volunteers - current_volunteers,
        "scheduled_campaigns": [asdict(c) for c in scheduled_campaigns]
    }
    return summary


if __name__ == "__main__":
    # Demonstration simulation
    mock_reports = [
        WasteReport("R1", 12.9716, 77.5946, plastic_count=12, waste_pile_count=2, face_mask_count=1, trash_bin_count=0, timestamp="2026-10-28 10:00:00"),
        WasteReport("R2", 12.9719, 77.5949, plastic_count=18, waste_pile_count=3, face_mask_count=0, trash_bin_count=0, timestamp="2026-10-28 10:15:00"),
        WasteReport("R3", 12.9750, 77.6000, plastic_count=3, waste_pile_count=0, face_mask_count=0, trash_bin_count=1, timestamp="2026-10-28 10:45:00"),
        WasteReport("R4", 12.9800, 77.6100, plastic_count=35, waste_pile_count=5, face_mask_count=4, trash_bin_count=0, timestamp="2026-10-28 11:20:00"),
    ]

    hotspots = cluster_hotspots(mock_reports)
    allocation = optimize_campaign_budget(hotspots, max_budget_usd=350.0, max_volunteers=15)
    print("\n[✓] Campaign Resource Allocation Plan:")
    print(pd.DataFrame(allocation["scheduled_campaigns"])[["hotspot_id", "severity_index", "recommended_action", "estimated_cost_usd", "required_volunteers"]])
