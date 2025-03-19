"""
Haversine Distance Calculation and Rectangle Area Generation Tool

Features:
1. Calculate the Haversine distance between two points (in kilometers)
2. Generate the four corner coordinates of a square area based on a center point and distance

Functions:
    haversine(lat1, lon1, lat2, lon2):
        Calculate the distance between two points (in kilometers)
        Parameters:
            lat1, lon1: Latitude and longitude of the first point
            lat2, lon2: Latitude and longitude of the second point
        Returns:
            Distance (in kilometers)

    calculate_rectangle(center_lat, center_lon, distance_km):
        Calculate the four corner coordinates of a square area
        Parameters:
            center_lat: Latitude of the center point
            center_lon: Longitude of the center point
            distance_km: Side length of the square (in kilometers)
        Returns:
            Dictionary of the four corner coordinates {
                "top_left": (lat, lon),
                "top_right": (lat, lon),
                "bottom_left": (lat, lon),
                "bottom_right": (lat, lon)
            }

Notes:
    1. Takes Earth's curvature into account
    2. Iterative method ensures accurate distance in the longitude direction
    3. Input and output are in decimal degrees
"""

import math


def haversine(lat1, lon1, lat2, lon2):
    # Earth's average radius (km)
    R = 6371.0
    # Convert degrees to radians
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def calculate_rectangle(center_lat, center_lon, distance_km):
    # Start with an approximate latitude/longitude offset
    R = 6371.0  # Earth's radius in km
    delta_lat = (distance_km / 2) / R  # Approximate latitude offset in radians
    delta_lat_deg = math.degrees(delta_lat)

    # Calculate longitude offset iteratively to match exact haversine distance
    def find_lon_offset(center_lat, center_lon, distance_km):
        low, high = 0, 180  # Search range for longitude offset in degrees
        while high - low > 1e-6:  # Precision threshold
            mid = (low + high) / 2
            candidate_lon = center_lon + mid
            dist = haversine(center_lat, center_lon, center_lat, candidate_lon)
            if dist < distance_km / 2:
                low = mid
            else:
                high = mid
        return low

    # Compute longitude offset
    delta_lon_deg = find_lon_offset(center_lat, center_lon, distance_km)

    # Calculate the rectangle's corners
    top_left = (center_lat + delta_lat_deg, center_lon - delta_lon_deg)
    top_right = (center_lat + delta_lat_deg, center_lon + delta_lon_deg)
    bottom_left = (center_lat - delta_lat_deg, center_lon - delta_lon_deg)
    bottom_right = (center_lat - delta_lat_deg, center_lon + delta_lon_deg)

    return {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right,
    }


def main():
    # Example usage
    center_lat = 48.8566  # Center latitude (e.g., Paris)
    center_lon = 2.3522  # Center longitude (e.g., Paris)
    distance_km = 3  # Desired square side length in km

    rectangle_coords = calculate_rectangle(center_lat, center_lon, distance_km)

    print("Rectangle corner coordinates based on haversine distance:")
    for corner, coords in rectangle_coords.items():
        print(f"{corner}: Latitude {coords[0]:.6f}, Longitude {coords[1]:.6f}")


if __name__ == "__main__":
    main()
