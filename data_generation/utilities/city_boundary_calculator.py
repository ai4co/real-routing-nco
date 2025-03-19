"""
Tool for calculating city boundary coordinates

This program calculates the boundary coordinates of a 3x3 square kilometer area around a specified city.
It automatically retrieves the city's center point coordinates and then calculates the boundary coordinates of this area.
The calculated data will be saved or updated in the city_bounding_coordinates.json file.

Usage:
    python -m data_generation.utilities.cal_data "city_name"

Parameters:
    city_name: The name of the city to process, e.g., "Beijing", "Shanghai", "New_York"
               For city names with spaces, use underscores instead

Examples:
    python -m data_generation.utilities.cal_data "Beijing"
    python -m data_generation.utilities.cal_data "New_York"

Notes:
    1. It is recommended to use English for city names
    2. Use underscores for city names with spaces
    3. The program will automatically merge or update existing data
"""

import argparse
import json
import os

import osmnx as ox

from data_generation.utilities import calculate_rectangle


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the boundary coordinates of a specified city"
    )
    parser.add_argument("city", type=str, help="name of the city to process")
    args = parser.parse_args()

    # Distance in kilometers for the bounding rectangle
    distance_km = 3

    # Initialize the result dictionary
    city_data = {}

    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, "city_bounding_coordinates.json")

    # Load existing data from city_bounding_coordinates.json
    existing_data = {}
    try:
        with open(json_file_path, "r") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        print("city_bounding_coordinates.json not found. A new file will be created.")

    try:
        city = args.city
        print(f"Processing city: {city}")
        # Get the center point of the city
        center_point = ox.geocode(city)

        # Calculate the bounding rectangle coordinates
        rectangle_coords = calculate_rectangle(
            center_point[0], center_point[1], distance_km
        )

        # Save data in the dictionary
        city_data[city] = {
            "center_point": {"latitude": center_point[0], "longitude": center_point[1]},
            "bounding_rectangle": rectangle_coords,
        }
    except Exception as e:
        print(f"Error processing city {city}: {e}")
        return

    # Merge new city data with existing data
    existing_data.update(city_data)

    # Save merged data back to city_bounding_coordinates.json
    with open(json_file_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    print("Data merged and saved to city_bounding_coordinates.json")


if __name__ == "__main__":
    main()
