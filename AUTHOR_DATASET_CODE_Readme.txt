This readme file was generated on [2023-11-21] by [Yu Han]

-------------------- GENERAL INFORMATION -------------------- 

Title of Dataset: Supplementary dataset and codes for "Classification of material and surface roughness using polarimetric multispectral LiDAR"

Name: Yu Han 
ORCID: 0000-0002-7290-3298
Institution: Institute of Geodesy and Photogrammetry
Address: ETH Zurich,8093 Zurich, Switzerland
Email: yu.han@geod.baug.ethz.ch

Name: David Salido Monzú
ORCID: 0000-0003-4274-6874
Institution: Institute of Geodesy and Photogrammetry 
Address: ETH Zurich,8093 Zurich, Switzerland
Email: david.salido@geod.baug.ethz.ch

Name: Andreas Wieser
ORCID: 0000-0001-5804-2164
Institution: Institute of Geodesy and Photogrammetry
Address: ETH Zurich,8093 Zurich, Switzerland
Email: andreas.wieser@geod.baug.ethz.ch


Date of data collection: 2023-03-29

Geographic location of data collection: ETH Zurich,8093 Zurich, Switzerland

Information about funding sources that supported the collection of the data: 184988 - Augmented Capability EDM using Phase and Power Spectral Signatures (SNF)


-------------------- SHARING/ACCESS INFORMATION -------------------- 

Licenses/restrictions placed on the data: Creative Commons Attribution-NonCommercial 4.0 International


Licenses/restrictions placed on the codes(scripts): MIT License
Copyright (c) 2023 Yu Han
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



Links to publications that cite or use the data: https://doi.org/10.1117/1.OE.62.11.114104



Recommended citation for this dataset: 
Yu Han, David Salido-Monzú, Andreas Wieser, "Classification of material and surface roughness using polarimetric multispectral LiDAR," Opt. Eng. 62(11) 114104 (21 November 2023)

--------------------   DATA & FILE OVERVIEW -------------------- 

File List: 

-->> Data processing and classification script: classification_surface_roughness_SPIE-OE_Finals.py

-->> Folder: Data
Raw spectra of the ten material specimens.

* 5 materials
PP (PP_pink)
PE (PE_red)
PVC (PVC_red)
Sandstone (sandstone_green)
Limestone (limestone)


* 2 levels of surface roughness
rougher (P80)
smoother (P400)

* 4 feature types
Standard reflectance spectrum (R_normal_SR)
Polarized reflectance spectrum (R_pol)
Unpolarized reflectance spectrum (R_unpol)
Degree of linear polarization spectrum (DoLP)

* Data structure (40*20)
40: 7 spectral channels with 40 nm bandwidth + 33 spectral channels with 10 nm bandwidth
20: 20 measurements at different surface positions per specimen


-->> Folder: Results_Features&Classification
Figures under the correpsonding paper.

* Features of the 10 material specimens (spectral configuration with 40 nm bandwidth)
Standard reflectance spectra (Fig_R[BW=40nm].png)
Polarized reflectance spectra (Fig_R_pol[BW=40nm].png)
Unpolarized reflectance spectra (Fig_R_unpol[BW=40nm].png)
Degree of linear polarization spectra (Fig_DoLP[BW=40nm].png)

* Features of the 10 material specimens (spectral configuration with 10 nm bandwidth)
Standard reflectance spectra (Fig_R[BW=10nm].png)
Polarized reflectance spectra (Fig_R_pol[BW=10nm].png)
Unpolarized reflectance spectra (Fig_R_unpol[BW=10nm].png)
Degree of linear polarization spectra (Fig_DoLP[BW=10nm].png)

* Accuracy scores of material classification 
Spectral configuration with 40 nm bandwidth (Fig_Material classification[BW=40nm].png)
Spectral configuration with 10 nm bandwidth (Fig_Material classification[BW=10nm].png)

* Accuracy scores of roughness classification 
Spectral configuration with 40 nm bandwidth (Fig_Roughness classification[BW=40nm].png)
Spectral configuration with 10 nm bandwidth (Fig_Roughness classification[BW=10nm].png)



