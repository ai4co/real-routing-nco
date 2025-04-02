import argparse
import datetime
import json
import os

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

from shapely.geometry import LineString, Polygon
from tqdm import tqdm

# path to save data
BASE_DATA_DIR = "../data_generation/data/city"
os.makedirs(BASE_DATA_DIR, exist_ok=True)


def clean_columns(gdf):
    gdf.columns = gdf.columns.str.lower()
    return gdf.loc[:, ~gdf.columns.duplicated()]


# Store cache by city
def get_city_cache_dir(city, timestamp=None):
    """
    Get the cache directory for the specified city. If a timestamp is provided, create a separate subdirectory.
    """
    city_dir = os.path.join(BASE_DATA_DIR, city.replace(" ", "_"))
    if timestamp:
        city_dir = os.path.join(city_dir, timestamp)
    os.makedirs(city_dir, exist_ok=True)
    return city_dir


# get city data and save to cache
def get_city_data_with_cache(city, bounding_coordinates_path, timestamp=None):
    """
    Store cached data by city or run timestamp.
    """
    with open(bounding_coordinates_path, "r") as f:
        city_bounding_data = json.load(f)

    city_cache_dir = get_city_cache_dir(city, timestamp)
    roads_cache = os.path.join(city_cache_dir, "roads.gpkg")
    water_cache = os.path.join(city_cache_dir, "water.gpkg")

    if os.path.exists(roads_cache) and os.path.exists(water_cache):
        print(f"Loading cached data for {city}...")
        filtered_roads = gpd.read_file(roads_cache, layer="filtered_roads")
        removed_roads = gpd.read_file(roads_cache, layer="removed_roads")
        water_features = gpd.read_file(water_cache)
        water_buffer = water_features.buffer(50)  # Recreate buffer
    else:
        print(f"Fetching data for {city}...")

        city_data = city_bounding_data.get(city)
        center_point = city_data["center_point"]
        print(f"Center point: {center_point}")

        bounding_rectangle = city_data["bounding_rectangle"]

        # Use bounding box to download map data
        north = bounding_rectangle["top_left"][0]
        south = bounding_rectangle["bottom_left"][0]
        west = bounding_rectangle["top_left"][1]
        east = bounding_rectangle["top_right"][1]

        bbox = (west, south, east, north)  # (left, bottom, right, top)

        print(f"Bounding box: {bbox}")

        print(f"Downloading map data for {city}...")
        graph = ox.graph_from_bbox(
            bbox,
            network_type="drive",
            simplify=True,
            retain_all=False,
            truncate_by_edge=False,
        )
        # 转换为 GeoDataFrame
        gdf_roads = ox.graph_to_gdfs(graph, nodes=False, edges=True)

        # Filter out bridges, tunnels, and motorways if those columns exist
        filtered_roads = gdf_roads.copy()

        if "bridge" in gdf_roads.columns:
            filtered_roads = filtered_roads[~filtered_roads["bridge"].notna()]

        if "tunnel" in gdf_roads.columns:
            filtered_roads = filtered_roads[~filtered_roads["tunnel"].notna()]

        if "highway" in gdf_roads.columns:
            filtered_roads = filtered_roads[filtered_roads["highway"] != "motorway"]

        # print("Downloading water features...")
        # tags = {"natural": ["water"], "waterway": True}
        # water_features = ox.features_from_place(city, tags=tags)

        # new methods
        print("Downloading water features...")
        try:
            # Fetch water features
            water_features = ox.features_from_bbox(
                north=north,
                south=south,
                east=east,
                west=west,
                tags={"natural": ["water"], "waterway": True},
            )

            if water_features is None or water_features.empty:
                raise ValueError("No water features found")

        except Exception as e:
            print(f"Warning: Could not fetch water features for {city}: {str(e)}")
            print("Creating empty water features DataFrame")
            # Create an empty GeoDataFrame
            water_features = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # make sure the water features are a GeoDataFrame
        if not isinstance(water_features, gpd.GeoDataFrame):
            print("Converting water features to GeoDataFrame")
            water_features = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        if water_features.crs is None:
            water_features.set_crs("EPSG:4326", inplace=True)

        water_features = water_features[water_features.geometry.is_valid]

        if len(water_features) == 0:
            water_features = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        print(f"Retrieved {len(water_features)} water features")

        # change crs to 3857
        filtered_roads = filtered_roads.to_crs(epsg=3857)
        water_features = water_features.to_crs(epsg=3857)

        print("water buffer")
        water_buffer = water_features.buffer(50)

        print("processing removed_roads and filtered_roads")
        removed_roads = filtered_roads[
            filtered_roads.intersects(water_buffer.unary_union)
        ]
        filtered_roads = filtered_roads[
            ~filtered_roads.intersects(water_buffer.unary_union)
        ]

        # save to cache
        print(f"Caching data for {city}...")
        filtered_roads.to_file(roads_cache, layer="filtered_roads", driver="GPKG")
        removed_roads.to_file(roads_cache, layer="removed_roads", driver="GPKG")
        water_features = clean_columns(water_features)
        water_features.to_file(water_cache, driver="GPKG")

    return filtered_roads, removed_roads, water_features, water_buffer


def generate_random_points(filtered_roads, area_of_interest, n_points=100, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # calculate the weight of each road
    filtered_roads["length"] = filtered_roads.geometry.length
    total_length = filtered_roads["length"].sum()
    filtered_roads["weight"] = filtered_roads["length"] / total_length

    # generate random points
    def batch_generate_points(roads, n_points, area):
        points = []
        batch_size = min(1000, n_points * 2)  # number in each batch

        with tqdm(total=n_points, desc="Generating random points") as pbar:
            while len(points) < n_points:
                # select roads based on weight
                selected_roads = roads.sample(
                    n=batch_size, weights="weight", replace=True
                )

                # Batch generate random distances
                random_distances = (
                    np.random.uniform(0, 1, batch_size) * selected_roads["length"].values
                )

                # check the point
                for road, distance in zip(selected_roads.geometry, random_distances):
                    if isinstance(road, LineString):
                        point = road.interpolate(distance)
                        if area.contains(point):
                            points.append(point)
                            pbar.update(1)
                            if len(points) >= n_points:
                                return points[:n_points]

        return points[:n_points]

    random_points = batch_generate_points(
        filtered_roads, n_points, area_of_interest
    )  # toGeoDataFrame
    gdf_points = gpd.GeoDataFrame(geometry=random_points, crs=filtered_roads.crs)

    return gdf_points


def get_area_of_interest(city, bounding_coordinates_path):
    # read bounding data
    with open(bounding_coordinates_path, "r") as f:
        city_bounding_data = json.load(f)

    # get the bounding data for the city
    city_data = city_bounding_data.get(city)
    if not city_data:
        raise ValueError(f"No bounding data found for {city}")

    bounding_rectangle = city_data["bounding_rectangle"]

    # create a polygon from the bounding rectangle
    area_of_interest = Polygon(
        [
            (
                bounding_rectangle["top_left"][1],
                bounding_rectangle["top_left"][0],
            ),  # (lon, lat)
            (bounding_rectangle["top_right"][1], bounding_rectangle["top_right"][0]),
            (
                bounding_rectangle["bottom_right"][1],
                bounding_rectangle["bottom_right"][0],
            ),
            (bounding_rectangle["bottom_left"][1], bounding_rectangle["bottom_left"][0]),
            (
                bounding_rectangle["top_left"][1],
                bounding_rectangle["top_left"][0],
            ),  # connect to the first point
        ]
    )

    # Project to EPSG:3857 coordinate system
    area_of_interest_projected = (
        gpd.GeoSeries([area_of_interest], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    )

    return area_of_interest_projected


def save_points_to_csv(gdf_points, filename):
    # convert to latlon
    gdf_points_latlon = gdf_points.to_crs(epsg=4326)
    sampled_coordinates = gdf_points_latlon.geometry.apply(
        lambda point: (point.y, point.x)
    ).tolist()

    df_coordinates = pd.DataFrame(sampled_coordinates, columns=["latitude", "longitude"])

    df_coordinates.to_csv(filename, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Collect usable points from the specified city"
    )
    parser.add_argument(
        "city",
        type=str,
        help="name of the ciyt, such as: 'Daejeon Metropolitan City, South Korea'",
    )
    parser.add_argument(
        "bounding_coordinates_path",
        type=str,
        help="path to the city bounding coordinates JSON file",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Create a timestamped cache for the current run",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility", default=None
    )
    args = parser.parse_args()

    city = args.city
    bounding_coordinates_path = args.bounding_coordinates_path
    seed = args.seed

    timestamp = None
    if args.timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    (
        filtered_roads,
        removed_roads,
        water_features_projected,
        water_buffer,
    ) = get_city_data_with_cache(city, bounding_coordinates_path, timestamp)

    area_of_interest_projected = get_area_of_interest(city, bounding_coordinates_path)

    gdf_points = generate_random_points(
        filtered_roads, area_of_interest_projected, n_points=100, seed=seed
    )

    # Save points to timestamped cache or path
    if timestamp:
        save_points_to_csv(
            gdf_points, f"{get_city_cache_dir(city, timestamp)}/sampled_coordinates.csv"
        )
    else:
        save_points_to_csv(
            gdf_points, f"../data/sampled_coordinates/{city}_sampled_coordinates.csv"
        )


if __name__ == "__main__":
    main()
