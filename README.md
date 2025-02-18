#  Discharge Estimation in Western US

Main project repository for UC Berkeley Fall 2022 Capstone project. 

__See the final presentation deck summarizing findings [here](https://docs.google.com/presentation/d/1pDesMmo2YIg6fW_5xN-YbuV7-70wK_8RJSI0UGFcpZ4/edit#slide=id.p).__

## Team

* Toby Petty
* Yao Chen
* Zixi Wang
* Kevin Xuan
* Anand Eyunni

## Contents

* __Data__: Data files
* __EDA__: Exploratory Data Analysis on outcome and input variables.

## Useful links

### Class documents

* [Class projects doc to keep up to date](https://docs.google.com/document/d/1LBM29ygvnNxYweuMhWh08fNtYtXCTP49gnXhdLb99wQ/edit?pli=1)
* [Week 4 team project plan](https://docs.google.com/document/d/1Qak9mXSkFQAeHKkSVHOWZ2kBlKRn31ycUrY6lt2gMjo/edit#)
* [Deliverables schedule](https://docs.google.com/document/d/1CWLP_c4wEdrYaAs8mT3jkurGmQO50WAOvU2H-OGuPIU/edit)
* [Class readings](https://docs.google.com/document/d/1KXirHpMz3D0WAwnwFvdO07wTHc-pEa3-EGJJQ5yrFiw/edit#heading=h.k5pp898cixi4)

### Streamgages
* [11185500](https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no=11185500)
* [11189500](https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no=11189500)
* [11202710](https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no=11202710)
* [11208000](https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no=11208000)
* [11266500](https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no=11266500)
* [11318500](https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no=11318500)
* [11402000](https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no=11402000)

### Hydrology

* [_Introduction to Hydrology_ textbook](https://margulis-group.github.io/teaching/)
* [How Streamflow is Measured](https://www.usgs.gov/special-topics/water-science-school/science/how-streamflow-measured)
* [What is streamflow data used for?](https://www.usgs.gov/special-topics/water-science-school/science/uses-streamflow-information)
* [USGS Water Data for the Nation](https://waterdata.usgs.gov/nwis/)
* [What Are Hydrologic Units?](https://water.usgs.gov/GIS/huc.html)
* [USGS National Hydrography Products](https://www.usgs.gov/national-hydrography/access-national-hydrography-products)
* [USGS State Shapefiles](https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Hydrography/NHD/State/Shape/)
* [How/Why Does the USGS Collect Streamflow Data](https://www.usgs.gov/centers/dakota-water-science-center/howwhy-does-usgs-collect-streamflow-data)

### Satellite data

* [ERA5-Land Hourly - ECMWF Climate Reanalysis](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY#bands)
* [MOD16A2.006: Terra Net Evapotranspiration 8-Day Global 500m](https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD16A2)
* [NASA SRTM Digital Elevation 30m](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003)
* [Google doc of resources from TA Colorado Reed](https://docs.google.com/document/d/1UYxjAyhIkgTUiOCvRwsWo-JBV9y0jmHluC0zWqU5M-Q/edit)
* [Professors' list of resources](https://docs.google.com/document/d/1f5q4R9SzMGcB734Oyqxy1ErVIpIZs1eCkHz0uaiqY4c/edit)
* [Zoom video recording with TA Colorado Reed](https://zoom.us/rec/play/LeGfgLJw4m33wnZFp5kqTWBikPFPYOAUOWMY62D2cGHgEOc5kE_5jd4ADvH9E4V3AacB9WWEOY1-jn8e.wMqvTPDgHHiK_d3o)
* [What is Remote Sensing?](https://www.earthdata.nasa.gov/learn/backgrounders/remote-sensing)
* [OpenET](https://openetdata.org/)
* [ERA5 hourly data on single levels from 1959 to present](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)

### Related work

* [Snowpack Estimation in Key Mountainous Water Basins from Openly-Available, Multimodal Data Sources](https://arxiv.org/pdf/2208.04246.pdf)
* [Towards Modeling and Predicting Water Table Levels in California](https://github.com/eddie-a-salinas/CA-Hydrology-W210-SP2022/blob/main/Towards%20Modeling%20and%20Predicting%20Water%20Table%20Levels%20in%20California.pdf)
* [Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks](https://hess.copernicus.org/articles/22/6005/2018/)
* [TorchGeo](https://github.com/microsoft/torchgeo)
* [BAIR Climate Initiative](https://data.berkeley.edu/news/new-uc-berkeley-initiative-uses-ai-research-solve-climate-problems)
* [BAIR Climate Initiative - MetaEarth](https://github.com/bair-climate-initiative/metaearth)
* [Driven Data Snowcast Challenge](https://drivendata.co/blog/swe-winners)

### Tools

* [GeoPandas - Plotting polygons with Folium](https://geopandas.org/en/stable/gallery/polygon_plotting_with_folium.html)
* [pysheds - Python library for calculating watersheds](https://github.com/mdbartos/pysheds)
* [Transformers in Time Series](https://github.com/qingsongedu/time-series-transformers-review)
* **Google Earth Engine**
  * [Python API](https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api)
  * [Service Account](https://developers.google.com/earth-engine/guides/service_account)
  * [Projections](https://developers.google.com/earth-engine/guides/projections)
  * [Download Image Collection to Drive](https://stackoverflow.com/questions/67289225/download-image-from-google-earth-engine-imagecollection-to-drive)
* [Working with GeoTIFF files](https://towardsdatascience.com/reading-and-visualizing-geotiff-images-with-python-8dcca7a74510)
* [USGS Drainage Area Identification](https://streamstats.usgs.gov/ss/)
* [Keras Deep Learning](https://github.com/fchollet/deep-learning-with-python-notebooks)
