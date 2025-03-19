#!/bin/bash

# Check command line parameters
if [ $# -lt 1 ]; then
    echo "Usage: $0 <city name> [clean]"
    echo "Example: $0 Beijing"
    echo "         $0 Beijing clean  # Clean existing services and files"
    exit 1
fi

CITY=$1
CONFIG_FILE="../data_generation/utilities/city_bounding_coordinates.json"
PORT=5000
PROJECT_ROOT=".."

check_and_add_city_boundary() {
    local city=$1
    local json_file=$2

    # Check if the city exists in JSON file
    if ! jq -e --arg city "$city" '.[$city]' "$json_file" > /dev/null; then
        echo "City $city not in config file, calculating boundary coordinates..."

        # Get Python script path
        PYTHON_SCRIPT_PATH="$PROJECT_ROOT/data_generation/utilities/city_boundary_calculator.py"

        if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
            echo "Boundary calculation script not found: $PYTHON_SCRIPT_PATH"
            return 1
        fi

        # Call Python script to calculate boundary
        if python3 "$PYTHON_SCRIPT_PATH" "$city"; then
            echo "Successfully added boundary coordinates for $city to config file"
            return 0
        else
            echo "Failed to add boundary coordinates for $city"
            return 1
        fi
    fi
    return 0
}

cleanup() {
    echo "Starting cleanup..."
    echo "Current working directory: $PWD"

    # Check initial port status
    echo "Checking port $PORT initial status..."
    check_port $PORT

    # Find and stop all Docker containers using this port
    container_id=$(sudo docker ps | grep ":$PORT->" | awk '{print $1}')
    if [ ! -z "$container_id" ]; then
        echo "Found container using port $PORT: $container_id"
        echo "Stopping container..."
        sudo docker stop $container_id
        echo "Removing container..."
        sudo docker rm $container_id
    else
        echo "No Docker containers found using port $PORT"
    fi

    # Check port status again
    echo "Checking port $PORT status after cleanup..."
    if check_port $PORT; then
        echo "Warning: Port $PORT is still in use"
        echo "Checking processes using the port..."
        sudo lsof -i :$PORT
    else
        echo "Port $PORT successfully released"
    fi

    # Delete all related files
    echo "Deleting all ${CITY} related files..."

    # List files to be deleted
    echo "The following files will be deleted:"
    ls -l ${CITY}* 2>/dev/null

    # Delete files
    rm -f ${CITY}*

    echo "Checking for remaining files..."
    remaining_files=$(ls ${CITY}* 2>/dev/null)
    if [ ! -z "$remaining_files" ]; then
        echo "Warning: The following files could not be deleted:"
        echo "$remaining_files"
    else
        echo "Cleanup complete, no remaining files!"
    fi

    # Final status report
    echo "Cleanup completion status:"
    echo "- File cleanup: $(ls ${CITY}* 2>/dev/null > /dev/null && echo "Files remaining" || echo "Completely cleaned")"
    echo "- Port status: $(check_port $PORT > /dev/null && echo "Still in use" || echo "Released")"

    exit 0
}

# Function to check port status
check_port() {
    local port=$1
    if sudo lsof -i :$port > /dev/null; then
        echo "Port $port is in use"
        return 0
    else
        echo "Port $port is not in use"
        return 1
    fi
}

# If second parameter is clean, execute cleanup
if [ "$2" = "clean" ]; then
    cleanup
fi

# Check dependencies
if ! command -v jq &> /dev/null; then
    echo "jq is required, please run: sudo apt-get install jq"
    exit 1
fi

# Check configuration file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Check and add city boundary coordinates
if ! check_and_add_city_boundary "$CITY" "$CONFIG_FILE"; then
    echo "Unable to get boundary coordinates for $CITY"
    exit 1
fi

# Check again if city exists in config
if ! jq -e ".[\"$CITY\"]" "$CONFIG_FILE" > /dev/null; then
    echo "City not found in config file: $CITY"
    echo "Available cities are:"
    jq -r 'keys[]' "$CONFIG_FILE"
    exit 1
fi

# Check if port is in use
if sudo docker ps | grep ":$PORT->" > /dev/null; then
    echo "Port $PORT is in use, attempting cleanup..."
    cleanup
fi

# Get bounding box coordinates
bottom_lat=$(jq -r ".[\"$CITY\"].bounding_rectangle.bottom_left[0]" "$CONFIG_FILE")
left_lon=$(jq -r ".[\"$CITY\"].bounding_rectangle.bottom_left[1]" "$CONFIG_FILE")
top_lat=$(jq -r ".[\"$CITY\"].bounding_rectangle.top_left[0]" "$CONFIG_FILE")
right_lon=$(jq -r ".[\"$CITY\"].bounding_rectangle.top_right[1]" "$CONFIG_FILE")

echo "Starting to process city: $CITY"
echo "Boundary range: $bottom_lat,$left_lon,$top_lat,$right_lon"

# Download data
echo "Downloading $CITY data..."
wget -O "${CITY}.osm" --post-data="[out:xml][timeout:300];
(
    way[\"highway\"](${bottom_lat},${left_lon},${top_lat},${right_lon});
    >;
);
out body;" https://overpass-api.de/api/interpreter

# Check download
if [ ! -f "${CITY}.osm" ]; then
    echo "Data download failed!"
    exit 1
fi

echo "Converting to PBF format..."
osmium cat "${CITY}.osm" -o "${CITY}.osm.pbf"
rm "${CITY}.osm"

if [ ! -f "${CITY}.osm.pbf" ]; then
    echo "PBF conversion failed!"
    exit 1
fi

# OSRM processing
echo "Processing PBF data..."
echo "1/3 Running osrm-extract..."
sudo docker run -t -v $(pwd):/data osrm/osrm-backend osrm-extract -p /opt/car.lua "/data/${CITY}.osm.pbf"

echo "2/3 Running osrm-partition..."
sudo docker run -t -v $(pwd):/data osrm/osrm-backend osrm-partition "/data/${CITY}.osrm"

echo "3/3 Running osrm-customize..."
sudo docker run -t -v $(pwd):/data osrm/osrm-backend osrm-customize "/data/${CITY}.osrm"

# Start service
echo "Starting OSRM service..."
sudo docker run -d -p ${PORT}:5000 -v $(pwd):/data osrm/osrm-backend osrm-routed --max-table-size 1000 --algorithm mld "/data/${CITY}.osrm"

# Wait for service to start
echo "Waiting for service to start..."
sleep 3  # Give the service 3 seconds to start

echo "Processing city data: $CITY"
PYTHON_SCRIPT_PATH="$PROJECT_ROOT/data_generation/utilities/create_dataset.py"

if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Data processing script not found: $PYTHON_SCRIPT_PATH"
    exit 1
fi

if python3 "$PYTHON_SCRIPT_PATH" "$CITY"; then
    echo "Successfully processed data for $CITY"

    # Get center point coordinates from config file (still needed for testing)
    center_lat=$(jq -r ".[\"$CITY\"].center_point.latitude" "$CONFIG_FILE")
    center_lon=$(jq -r ".[\"$CITY\"].center_point.longitude" "$CONFIG_FILE")
else
    echo "Failed to process data for $CITY"
    exit 1
fi

echo "Service started!"
echo "- Access address: http://localhost:${PORT}"
echo "- Test command: curl 'http://localhost:${PORT}/route/v1/driving/$center_lon,$center_lat;$center_lon,$center_lat'"
echo ""
echo "To clean up services and files, run:"
echo "$0 $CITY clean"
