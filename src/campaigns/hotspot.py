
from collections import defaultdict
from math import radians, sin, cos, sqrt, atan2


def haversine_km(lat1, lon1, lat2, lon2):
    earth_radius = 6371.0

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    return 2 * earth_radius * atan2(
        sqrt(a), sqrt(1 - a)
    )


def detect_hotspots(records):
    """
    Records require:
    site_id, latitude, longitude, detections
    """

    sites = defaultdict(lambda: {
        "site_id": None,
        "latitude": None,
        "longitude": None,
        "total_detections": 0,
        "waste_types": {},
    })

    for record in records:
        site_id = record.get("site_id")
        latitude = record.get("latitude")
        longitude = record.get("longitude")

        if not site_id or latitude is None or longitude is None:
            continue

        latitude = float(latitude)
        longitude = float(longitude)

        if not (-90 <= latitude <= 90):
            raise ValueError(f"Invalid latitude for {site_id}")

        if not (-180 <= longitude <= 180):
            raise ValueError(f"Invalid longitude for {site_id}")

        site = sites[site_id]

        if site["latitude"] is not None:
            if (
                site["latitude"] != latitude
                or site["longitude"] != longitude
            ):
                raise ValueError(
                    f"Conflicting coordinates for site {site_id}"
                )

        site["site_id"] = site_id
        site["latitude"] = latitude
        site["longitude"] = longitude

        for detection in record.get("detections", []):
            class_name = detection["class_name"]
            site["total_detections"] += 1
            site["waste_types"][class_name] = (
                site["waste_types"].get(class_name, 0) + 1
            )

    results = []

    for site in sites.values():
        if site["total_detections"] == 0:
            continue

        results.append({
            **site,
            "priority_score": site["total_detections"],
        })

    return sorted(
        results,
        key=lambda item: item["priority_score"],
        reverse=True,
    )


def nearby_hotspots(hotspots, radius_km=2.0):
    """Group nearby sites for inspection; does not merge their counts."""

    clusters = []
    remaining = list(hotspots)

    while remaining:
        center = remaining.pop(0)

        cluster = [center]

        for candidate in remaining[:]:
            distance = haversine_km(
                center["latitude"],
                center["longitude"],
                candidate["latitude"],
                candidate["longitude"],
            )

            if distance <= radius_km:
                cluster.append(candidate)
                remaining.remove(candidate)

        clusters.append(cluster)

    return clusters
