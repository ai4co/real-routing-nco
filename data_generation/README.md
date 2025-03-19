# Data Generation Usage Guide

## Directory Structure

```
data_generation/
├── data/
│   ├── sampled_coordinates/
│   ├── city/
│   ├── sample_points_csv/
│   └── matrix/
└── utilities/
    ├── create_dataset.py
    ├── haversine.py
    └── road_points_sampler.py

```




## OSRM Installation Instructions


We will install the [OSRM backend](https://github.com/Project-OSRM/osrm-backend) to provide routing services locally without any API!

<details close>
<summary>OSRM Installation Instructions</summary>

(Tested on Ubuntu 24.04)

Download map:

```bash
cd osrm/
wget https://download.geofabrik.de/europe/germany/Bolo-latest.osm.pbf
```
wget https://download.geofabrik.de/asia/south-korea-latest.osm.pbf

> Tip: You can download maps from https://download.geofabrik.de/. If you want to download  e.g. all of South Korea, you can download the file `asia/south-korea-latest.osm.pbf`.


Pull docker image:

```bash
docker pull ghcr.io/project-osrm/osrm-backend:v5.27.1
```

Run docker container the following commands, which will generate the files `bremen-latest.osrm.*` etc:

```bash
sudo docker run -t -v $(pwd):/data -v $(pwd):/data osrm/osrm-backend osrm-extract -p /opt/car.lua /data/bremen-latest.osm.pbf

sudo docker run -t -v $(pwd):/data -v $(pwd):/data osrm/osrm-backend osrm-partition /data/bremen-latest.osrm

sudo docker run -t -v $(pwd):/data -v $(pwd):/data osrm/osrm-backend osrm-customize /data/bremen-latest.osrm
```

Sourth Korea

```bash
sudo docker run -t -v $(pwd):/data -v $(pwd):/data osrm/osrm-backend osrm-extract -p /opt/car.lua /data/south-korea-latest.osm.pbf

sudo docker run -t -v $(pwd):/data -v $(pwd):/data osrm/osrm-backend osrm-partition /data/south-korea-latest.osrm

sudo docker run -t -v $(pwd):/data -v $(pwd):/data osrm/osrm-backend osrm-customize /data/south-korea-latest.osrm
```

</details>


Then, you can run the server with the preferred data file:

```bash
sudo docker run -t -i -p 5000:5000 -v $(pwd):/data osrm/osrm-backend osrm-routed --algorithm mld /data/bremen-latest.osrm --max-table-size 10000
```

```bash
sudo docker run -t -i -p 5000:5000 -v $(pwd):/data osrm/osrm-backend osrm-routed --algorithm mld /data/south-korea-latest.osrm --max-table-size 10000
```

Now, you can access the server at `http://localhost:5000/`.



## Usage

### How to Collect Data with OSRM

1. cd to the OSRM folder. This is important, as the docker image works with the current directory.
2. Run the following command:
   ```bash
   bash start_sample.sh <city_name>
   ```
Replace `<city_name>`with the name of the city.
Please pay attention to case sensitivity.
Only city names that satisfy the following query can be used:
```python
center_point = ox.geocode(city)
```
This command will collect 1000 data points within a 9 square kilometer area of the specified city.
The collected data will be saved in the `/real-routing-nco/data`.

### Clean up intermediate files

In the osrm folder, run the following command:
   ```bash
   bash start_sample.sh <city_name> clean
   ```
This can remove the files required to start the OSRM service in the OSRM directory.

### If you want to ganerate points from a list of cities:

1. cd to the OSRM folder. This is important, as the docker image works with the current directory.
2. Run the following command:
   ```bash
   bash 100cities.sh
   ```
Make sure that you have set basic parameters in the script.
