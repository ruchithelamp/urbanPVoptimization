# Optimizing Rooftop Solar Potential for a City with Machine Learning  
_MADS Fall 2025 Capstone Project_

## Project Overview

This project estimates how much of a city’s electricity demand could be met using **rooftop solar photovoltaic (PV) systems** on existing buildings.

We focus on two proof-of-concept cities:

- **Ann Arbor, Michigan** Experiences a Humid Continental Climate with mixed heating and cooling loads, strong winter peaks  
- **Tucson, Arizona** Experiences a hot and arid climate with high summer cooling loads  

Using high-resolution satellite imagery, this code will:

1. *Detect rooftops* with a Convolutional Neural Network (CNN).  
2. *Segment rooftop footprints* with a YOLO-based model trained on polygon labels.  
3. *Estimate PV panel capacity* per rooftop using typical panel dimensions and nameplate ratings.  
4. *Compare modeled PV generation* to historical hourly and monthly electricity demand to assess rooftop PV’s potential contribution to city load and load growth.

The repository contains all of the code that was used to train the models, generate the figures in the report, and produce an interactive tool for exploring rooftop PV potential.

You can view all dependencies in the [requirements.txt](site/requirements.txt) file.

If you have any issues running this code, or have questions for the creators of this project, please contact the following people:

Stephanie Maciejewski: `<PteroisRadiata>`, `<smacieje@umich.edu>`

Erin Mettler: `<Emettler97>` `<emettler@umich.edu>`,

Ruchi Patil: `<Ruchithelamp>` `<ruchipat@umich.edu>`,



# DATA ACCESS STATEMENT
### Demand Data 

Hourly demand data is collected from electric balancing authorities by the U.S. Energy Information Administration (EIA) via Form-930. Raw data is made available to the public through the EIA Open Data portal.

https://www.eia.gov/opendata/

The EIA began collecting hourly demand data in July of 2015 and continuously publishes new values each day. The reported demand value for each hour corresponds to the integrated mean value in megawatts over the previous hour. Cleaned and archived EIA data publicly available for download from:

Ruggles, T.H., Farnham, D.J., Tong, D. et al. Developing reliable hourly electricity demand data through screening and imputation. Sci Data 7, 155 (2020). https://doi.org/10.1038/s41597-020-0483-x

and the data archive:

Ruggles, Tyler H., Farnham, David J. & Wongel, Alicia (2025). EIA Cleaned Hourly Electricity Demand Data (Version v1.4) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3517196

<https://github.com/truggles/EIA_Cleaned_Hourly_Electricity_Demand_Data>

### Satellite Imagery

This repository uses Landsat imagery courtesy of the U.S. Geological Survey. Data is public domain (USGS).

<https://earthexplorer.usgs.gov/>

### Permit Data

This project includes public-record permit data from municipal websites:

- City of Ann Arbor — permit documents obtained from the Ann Arbor online permitting portal.
  https://stream.a2gov.org/energov_prod/selfservice/#/search
- City of Tucson — permit documents obtained from the Tucson Planning & Development Services website.
  https://www.tucsonaz.gov/Departments/Planning-Development-Services/Development-Tools-Resources/Development-Activity-Reports/Permit-Activity

All materials are public records.

"Licenses for data use and redistribution are respected."
