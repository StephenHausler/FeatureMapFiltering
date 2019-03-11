# FeatureMapFiltering
Source code to ACRA paper "Feature Map Filtering: Improving Visual Place Recognition with Convolutional Calibration".
If you use this source code, please cite the following reference:
Hausler, S., Jacobson A., and Milford, M. (2018). Feature Map Filtering: Improving Visual Place Recognition with Convolutional Calibration. In Australasian Conference on Robotics and Automation (ACRA).

To use this source code, recommend downloading the St Lucia dataset. Please go to https://wiki.qut.edu.au/display/cyphy/St+Lucia+Multiple+Times+of+Day and download "180809_1545" and "190809_0845". 

Then extract individual frames out of the downloaded videos, for example, using Avconv on Ubuntu (https://libav.org/avconv.html). 

Edit the file paths in the source codes to point to the save locations of your downloaded dataset. Please note, HybridNet is not included in this release, please either contact the original author of HybridNet, or replace with another network such as AlexNet or VGG-16.




Acknowledgements:

MATLAB Libaries: MATLAB;
sort_nat: Douglas M. Schwarz copyright 2008;
Hybrid Net (not included in this release): Zetao Chen 2017.

