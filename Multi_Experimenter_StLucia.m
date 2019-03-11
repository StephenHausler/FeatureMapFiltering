%--------------------------------------------------------------------------
%   Project Title: Feature Map Filtering
%   Author: Stephen Hausler
%   
%   Open Source Code, requires MATLAB with Neural Network Toolbox.
%   Refer to LICENSES.txt for license to this source code and 3RD_PARTY_
%   LICENSES for all 3rd party licences.
%-------------------------------------------------------------------------

clear variables
% Your neural network:
datafile = 'D:/MATLAB/CNN_feature_map_filter/HybridNet/HybridNet.caffemodel';
protofile = 'D:/MATLAB/CNN_feature_map_filter/HybridNet/deploy.prototxt';
net = importCaffeNetwork(protofile,datafile);
 
%net = alexnet;  %Alexnet trained on Imagenet challenge

actArray = [11 13 15 17];  %for Hybridnet (conv 3,4,5,6 ReLu layers)
%actArray = [6 11 13 15];   %for Alexnet
deltaArray = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];

%adjusting actArray changes the layer to be filtered.
%adjusting deltaArray changes the early stopping point.
pass = Feature_Map_Filter(actArray(1), deltaArray(1), net);

saveName = 'StLucia_finalMaps_HybridNet_earlyStop_01_actLayer_11.mat';
[recall,precision,AvTime,TotalTime] = Single_Frame_Place_Recognition(saveName,net,actLayer);

save('HybridNet_StLucia_FilterResults.mat','recall_array',...
    'precision_array','AvTime','TotalTime');

