import os

import folium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def render(
    points,
    center,
    zoom,
    temp_path,
    save_path,
):
    """
    Args:
        - points: (N, 2) numpy array
        - center: (2,) numpy array
        - zoom: int
        - temp_path: str, the path to save the map html file
        - save_path: str, the path to save the screenshot
    """
    # Init the map
    map = folium.Map(location=center, zoom_start=zoom)

    # Plot points
    for lat, lon in points:
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,  # Size of the circle
            color="blue",  # Color of the boundary
            fill=True,
            fillColor="blue",  # Color of the circle
            fillOpacity=0.7,
            popup=f"Location: ({lat:.4f}, {lon:.4f})",
        ).add_to(map)

    # Temp save the map to html
    map.save(temp_path)

    # Init the Chrome driver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    width = 1800  # Customize width
    height = 1800  # Customize height
    driver.set_window_size(width, height)

    file_path = os.path.abspath(temp_path)
    driver.get(f"file://{file_path}")

    # Wait for one second to load the map, otherwise the screenshot will be blank
    os.system("sleep 1")

    driver.save_screenshot(save_path)
    driver.quit()
