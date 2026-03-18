# Option 1: Direct download (2.3GB)
# wget https://microsoftbuildingfootprints.blob.core.windows.net/geojson/Nigeria.geojsonl.zip

# Option 2: Using Python
import urllib.request
urllib.request.urlretrieve(
    "https://microsoftbuildingfootprints.blob.core.windows.net/geojson/Nigeria.geojsonl.zip",
    "Nigeria.geojsonl.zip"
)

# Unzip
unzip Nigeria.geojsonl.zip