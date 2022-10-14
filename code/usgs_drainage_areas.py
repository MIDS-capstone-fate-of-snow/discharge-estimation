
import folium
import geopandas as gpd


class USGSDrainageArea:
    """Class for handling geojson files of streamgage drainage areas downloaded
    from USGS' streamstats service:

        https://streamstats.usgs.gov/ss/
    """

    def __init__(self, fp: str):
        self.gdf = gpd.read_file(fp)
        self.point = self.gdf[self.gdf["id"] == "globalwatershedpoint"]["geometry"].iloc[0]
        self.lon, self.lat = self.point.coords.xy[0][0], self.point.coords.xy[1][0]
        self.shape = self.gdf[self.gdf["id"] == "globalwatershed"]["geometry"].iloc[0]
        self.bounding_box = self.shape.bounds
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = self.bounding_box
        self.box_centroid = self.shape.centroid
        self.centroid_lon, self.centroid_lat = self.box_centroid.coords.xy[0][0], self.box_centroid.coords.xy[1][0]

        # Calculate area:
        geo_series = self.gdf["geometry"].to_crs({"proj": "cea"})
        self.area_m = geo_series.area.loc[1]
        self.area_km = self.area_m / 10 ** 6
        self.area_miles = self.area_km * 0.386102

    def fmap(self, zoom_start: int = 10, width="100%", height="100%",
             my_map=None):
        """Create a folium map of the drainage area."""
        if my_map is None:
            my_map = folium.Map(location=[self.centroid_lat, self.centroid_lon],
                                zoom_start=zoom_start, width=width, height=height)

        # Map the point:
        folium.Marker(location=[self.lat, self.lon], radius=10,
                      popup=f"lat={self.lat}, lon={self.lon}").add_to(my_map)

        # Map the basin shape:
        sim_geo = gpd.GeoSeries(self.shape).simplify(tolerance=0.001)
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {"fillColor": "orange"})
        geo_j.add_to(my_map)

        # Map the bounding box around the basin:
        folium.Rectangle([(self.min_lat, self.min_lon), (self.max_lat, self.max_lon)],
                         color="red", weight=2, fill=True, fill_opacity=0.15).add_to(my_map)

        return my_map
