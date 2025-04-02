import argparse
import json
import os

from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from shapely.geometry import Polygon

from data_generation.utilities import (
    generate_random_points,
    get_city_data_with_cache,
    save_points_to_csv,
)

SEED = 42


class PathManager:
    def __init__(self, project_root: str):
        self.project_root = project_root
        # store.npzfile
        self.data_dir = os.path.join(project_root, "data")
        # store gkgp file
        self.data_generation = os.path.join(project_root, "data_generation")
        self.data_generation_data = os.path.join(self.data_generation, "data")
        self.utilities_dir = os.path.join(self.data_generation, "utilities")

        # create necessary directory structure
        self._create_directory_structure()

    def _create_directory_structure(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.data_generation_data, exist_ok=True)
        os.makedirs(os.path.join(self.data_generation_data, "city"), exist_ok=True)

    def get_config_path(self) -> str:
        return os.path.join(self.utilities_dir, "city_bounding_coordinates.json")

    def get_sampled_coordinates_path(self, city_name: str) -> str:
        formatted_name = city_name.replace(" ", "_")
        sampled_coords_dir = os.path.join(
            self.data_generation_data, "sampled_coordinates"
        )
        os.makedirs(sampled_coords_dir, exist_ok=True)
        return os.path.join(
            sampled_coords_dir, f"{formatted_name}_sampled_coordinates.csv"
        )

    def get_city_data_path(self, city_name: str) -> str:
        # .npz path
        formatted_name = city_name.replace(" ", "_")
        city_folder = os.path.join(self.data_dir, formatted_name)
        os.makedirs(city_folder, exist_ok=True)
        return os.path.join(city_folder, f"{formatted_name}_data.npz")

    def get_city_gpkg_path(self, city_name: str) -> Dict[str, str]:
        # .gpkg path
        formatted_name = city_name.replace(" ", "_")
        city_folder = os.path.join(self.data_generation_data, "city", formatted_name)
        os.makedirs(city_folder, exist_ok=True)
        return {
            "road": os.path.join(city_folder, "roads.gpkg"),
            "water": os.path.join(city_folder, "waters.gpkg"),
        }


def split_points(points, chunk_size=100):
    for i in range(0, len(points), chunk_size):
        yield points[i : i + chunk_size]


class OSRMRouter:
    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url

    def get_table(
        self, points: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        coords = ";".join([f"{lon:.6f},{lat:.6f}" for lat, lon in points])
        url = f"{self.server_url}/table/v1/driving/{coords}"
        params = {"annotations": "distance,duration"}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data["code"] != "Ok":
                print(f"OSRM returned non-OK status: {data['code']}")
                return None, None

            distances = np.array(data["distances"])
            durations = np.array(data["durations"])

            distances = np.where(distances is None, 1e9, distances) / 1000
            durations = np.where(durations is None, 1e9, durations) / 60

            distances = np.where(distances < 0, 1e-3, distances)
            durations = np.where(durations < 0, 1e-3, durations)

            print("Successfully processed the full 1000x1000 matrix.")
            return distances, durations

        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            return None, None
        except (KeyError, TypeError, ValueError) as e:
            print(f"Data processing error: {e}")
            return None, None


def process_city_data(city_name: str, paths: PathManager) -> None:
    print(f"Processing city: {city_name}")

    with open(paths.get_config_path(), "r") as f:
        city_bounding_data = json.load(f)

    city_data = city_bounding_data.get(city_name)
    if not city_data:
        raise ValueError(f"No bounding data found for {city_name}")

    bounding_rectangle = city_data["bounding_rectangle"]
    print("Creating area of interest polygon...")
    area_of_interest = Polygon(
        [
            (bounding_rectangle["top_left"][1], bounding_rectangle["top_left"][0]),
            (bounding_rectangle["top_right"][1], bounding_rectangle["top_right"][0]),
            (
                bounding_rectangle["bottom_right"][1],
                bounding_rectangle["bottom_right"][0],
            ),
            (bounding_rectangle["bottom_left"][1], bounding_rectangle["bottom_left"][0]),
            (bounding_rectangle["top_left"][1], bounding_rectangle["top_left"][0]),
        ]
    )

    area_of_interest_projected = (
        gpd.GeoSeries([area_of_interest], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    )

    (
        filtered_roads,
        removed_roads,
        water_features,
        water_buffer,
    ) = get_city_data_with_cache(city_name, paths.get_config_path())
    gdf_points = generate_random_points(
        filtered_roads, area_of_interest_projected, n_points=1000, seed=SEED
    )

    sampled_coords_path = paths.get_sampled_coordinates_path(city_name)
    save_points_to_csv(gdf_points, sampled_coords_path)

    df = pd.read_csv(sampled_coords_path)
    points = list(zip(df["latitude"], df["longitude"]))

    router = OSRMRouter()
    distance_matrix, duration_matrix = router.get_table(points)

    if distance_matrix is not None and duration_matrix is not None:
        print("saving data...")
        npz_file = paths.get_city_data_path(city_name)
        try:
            np.savez_compressed(
                npz_file,
                distance=distance_matrix,
                duration=duration_matrix,
                points=np.array(points),
            )
            print(f"Data successfully saved to: {npz_file}")
        except Exception as e:
            print(f"Error saving file: {e}")
            print(f"Attempting to save to path: {npz_file}")
            print(
                f"Directory permissions: {os.access(os.path.dirname(npz_file), os.W_OK)}"
            )


def main():
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    print(f"Project root directory: {project_root}")

    paths = PathManager(project_root)

    parser = argparse.ArgumentParser(description="Process city data from a city name.")
    parser.add_argument("city_name", type=str, help="Name of the city to process.")
    args = parser.parse_args()

    process_city_data(args.city_name, paths)


if __name__ == "__main__":
    main()
