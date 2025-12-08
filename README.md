# Mapping Rooftop Solar Potential with Deep Learning  
_MADS Fall 2025 Capstone Project_

## Project Overview

This project estimates how much of a city’s electricity demand could be met using **rooftop solar photovoltaic (PV) systems** on existing buildings.

We focus on two proof-of-concept cities:

- **Ann Arbor, Michigan** Experiences a Humid Continental Climate with mixed heating and cooling loads, strong winter peaks  
- **Tucson, Arizona** Experiences a hot and arid climate with high summer cooling loads  

Using high-resolution satellite imagery, we:

1. *Detect rooftops* with a Convolutional Neural Network (CNN).  
2. *Segment rooftop footprints* with a YOLO-based model trained on polygon labels.  
3. *Estimate PV panel capacity* per rooftop using typical panel dimensions and nameplate ratings.  
4. *Compare modeled PV generation* to historical hourly and monthly electricity demand to assess rooftop PV’s potential contribution to city load and load growth.

The repository contains all of the code that was used to train the models, generate the figures in the report, and produce an interactive tool for exploring rooftop PV potential.This is an end-to-end pipeline that will be able to detect existing rooftops and PV panels from aerial sattelite images, compute the solar/PV panel potential of those rooftops, and export the optimal mix of rooftops to install PV panels on to meet a user-specific electric capacity amount.

If you have any issues running this code, or have questions for the creators of this project, please contact the following:
Erin Mettler: `<Emettler97>` `<emettler@umich.edu>`,
Ruchi Patil: `<Ruchithelamp>` `<ruchipat@umich.edu>`,
Stephanie Maciejewski: `<PteroisRadiata>`, `<smacieje@umich.edu>`
