#!/bin/bash
# Get sudo privileges at the start of the script
sudo -v

# Run a background process to periodically update the sudo timestamp
while true; do
    sudo -n true
    sleep 60
    kill -0 "$$" || exit
done 2>/dev/null &

# Set basic parameters
CONFIG_FILE="../data_generation/utilities/city_bounding_coordinates.json"
CITIES_FILE="../data_generation/100cities.json"
PORT=5000
CURRENT_DIR=$(pwd)

# Check dependencies
if ! command -v jq &> /dev/null; then
    echo "jq is required, please run: sudo apt-get install jq"
    exit 1
fi

# Check Docker permissions
if ! docker info > /dev/null 2>&1; then
    echo "Docker permissions are required, please run: sudo usermod -aG docker $USER"
    echo "Then re-login or restart the system"
    exit 1
fi

# Cleanup function
cleanup() {
    local city=$1
    local port=$2
    echo "Cleaning up services and files for $city..."

    # Stop and remove Docker container
    container_id=$(docker ps | grep ":$port->" | awk '{print $1}')
    if [ ! -z "$container_id" ]; then
        docker stop $container_id
        docker rm $container_id
    fi

    # Remove related files
    rm -f ${city}*
}

# Process a single city
process_city() {
    local city=$1
    echo "Starting to process city: $city"

    # Check if the result file already exists
    if [ -f "../data/${city}/${city}_data.npz" ]; then
        echo "Dataset for city ${city} already exists, skipping processing"
        return 0
    fi

    # Create city data directory
    mkdir -p "../data_generation/data/${city}"

    # Check and add city boundary coordinates
    if ! python3 ../data_generation/utilities/city_boundary_calculator.py "$city"; then
        echo "Unable to get boundary coordinates for $city"
        return 1
    fi

    # Get boundary coordinates
    bottom_lat=$(jq -r ".[\"$city\"].bounding_rectangle.bottom_left[0]" "$CONFIG_FILE")
    left_lon=$(jq -r ".[\"$city\"].bounding_rectangle.bottom_left[1]" "$CONFIG_FILE")
    top_lat=$(jq -r ".[\"$city\"].bounding_rectangle.top_left[0]" "$CONFIG_FILE")
    right_lon=$(jq -r ".[\"$city\"].bounding_rectangle.top_right[1]" "$CONFIG_FILE")

    # Download data
    echo "Downloading data for $city..."
    wget -O "${city}.osm" --post-data="[out:xml][timeout:300];
    (
        way[\"highway\"](${bottom_lat},${left_lon},${top_lat},${right_lon});
        >;
    );
    out body;" https://overpass-api.de/api/interpreter

    if [ ! -f "${city}.osm" ]; then
        echo "Failed to download data for ${city}"
        return 1
    fi

    # Convert to PBF format
    osmium cat "${city}.osm" -o "${city}.osm.pbf"
    rm "${city}.osm"

    if [ ! -f "${city}.osm.pbf" ]; then
        echo "Failed to convert data for ${city} to PBF format"
        return 1
    fi

    # OSRM processing
    echo "Processing PBF data..."
    docker run -t -v $(pwd):/data osrm/osrm-backend osrm-extract -p /opt/car.lua "/data/${city}.osm.pbf"
    docker run -t -v $(pwd):/data osrm/osrm-backend osrm-partition "/data/${city}.osrm"
    docker run -t -v $(pwd):/data osrm/osrm-backend osrm-customize "/data/${city}.osrm"

    # Start service
    docker run -d -p ${PORT}:5000 -v $(pwd):/data osrm/osrm-backend osrm-routed --max-table-size 1000 --algorithm mld "/data/${city}.osrm"

    # Wait for the service to start
    sleep 3

    # Process city data
    if python3 ../data_generation/utilities/create_dataset.py "$city"; then
        echo "Successfully processed data for $city"
    else
        echo "Failed to process data for $city"
        return 1
    fi

    # Clean up services and files for the current city
    cleanup "$city" "$PORT"

    echo "$city processing completed"
    return 0
}

# Main program
echo "Starting batch processing of cities..."

# Read city list
cities=$(jq -r '.subfolders[]' "$CITIES_FILE")

# Iterate and process each city
for city in $cities; do
    echo "==============================================="
    echo "Processing: $city"

    if process_city "$city"; then
        echo "Successfully processed $city"
    else
        echo "Failed to process $city"
        # Record failed cities
        echo "$city" >> failed_cities.txt
    fi

    # Pause before processing the next city
    sleep 5
done

echo "All cities processed"
if [ -f failed_cities.txt ]; then
    echo "The following cities failed to process:"
    cat failed_cities.txt
fi
