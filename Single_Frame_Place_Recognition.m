%--------------------------------------------------------------------------
%   Project Title: Feature Map Filtering
%   Author: Stephen Hausler
%   
%   Open Source Code, requires MATLAB with Neural Network Toolbox.
%   Refer to LICENSES.txt for license to this source code and 3RD_PARTY_
%   LICENSES for all 3rd party licences.
%-------------------------------------------------------------------------

function [recall,precision,AvTime,TotalTime] = Single_Frame_Place_Recognition(saveName,net,actLayer)

figure

%single frame VPR

load(saveName);
Dataset = 1;  %1 - Lucia or Oxford. 2 - Nordland.
skip = 0; %only set to 1 if already done reference traverse

%Reference Database File Address (provide full path to folder containing images or video):
Ref_folder = 'D:\Windows\St_Lucia_Dataset\0845_15FPS\Frames';
Ref_file_type = '.jpeg';     
Imstart_R = 790;  %start dataset after this many frames or seconds for video.
finalImage_R = 3900;

% Ref_folder = 'D:\Windows\Nordland\nordland_summer_images';
% Ref_file_type = '.png';     
% Imstart_R = 2000;  %start dataset after this many frames or seconds for video.
% finalImage_R = 4000;

%Query Database File Address (provide full path to folder containing images or video):
Query_folder = 'D:\Windows\St_Lucia_Dataset\1545_15FPS\Frames';
Query_file_type = '.jpeg';
Imstart_Q = 700;  %start dataset after this many frames or seconds for video.
finalImage_Q = 3700;

% Query_folder = 'D:\Windows\Nordland\nordland_winter_images';
% Query_file_type = '.png';     
% Imstart_Q = 2000;  %start dataset after this many frames or seconds for video.
% finalImage_Q = 4000;

% Ref_folder = 'D:\Windows\oxford-data\2014-12-09-13-21-02\stereo\left_rect';
% Ref_file_type = '.png';
% Imstart_R = 0;
% finalImage_R = 5500;

% Query_folder = 'D:\Windows\oxford-data\2014-12-10-18-10-50\stereo\left_rect';
% Query_file_type = '.png';
% Imstart_Q = 0;
% finalImage_Q = 6000;

Frame_skip = 3;   
%Frame_skip = 1;
Template_count = 0;

%Ground truth load:
%Load a ground truth correspondance matrix.
GT_file = load('D:\Windows\St_Lucia_Dataset\StLucia_GPSMatrix.mat');
%GT_file = load('D:\Windows\oxford-data\OxfordRobotCar_GPSMatrix.mat');

thresh = [0.0001 0.0005 0.001 0.0015 0.002 0.0025 0.003 0.0035 0.004 0.0045...
    0.005 0.0055 0.0065 0.007 0.0075 0.008 0.0085 0.009  0.0095 0.01 0.011...
    0.012 0.013 0.014 0.015 0.0175 0.02 0.025 0.03 0.035 0.04 0.045 0.05 ...
    0.06 0.07 0.08 0.09 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3...
    0.32 0.34 0.36 0.38 0.4 0.42 0.44 0.46 0.48 0.5 0.6 0.7 0.8 0.9 1.0]; 

obsThresh = 0.1;
epsilon = 0.001;
Rwindow = 20;           %Change this to reflect the approximate distance 
%between frames and the localisation accuracy required.
%Rwindow = 10;  %Nordland

%Image processing method adjustable settings:
Initial_crop = [20 60 0 0];  %crop amount in pixels top, bottom, left, right
Initial_crop = Initial_crop + 1;    %only needed because MATLAB starts arrays from index 1.
CNN_resize = [227 227]; %for HybridNet
%CNN_resize = [224 224]; %for VGG-16

%Zeroing Variables
recall_count = zeros(1,length(thresh));
error_count = zeros(1,length(thresh));
false_negative_count = zeros(1,length(thresh));
recall_count2 = 0;
error_count2 = 0;
truePositive = [0 0]; 
falsePositive = [0 0];
Template_count_for_plot = 0;
plot_skip = 1;

%Reference Dataset:
%--------------------------------------------------------------------------
Ref_file_type = strcat('*',Ref_file_type);
fR = dir(fullfile(Ref_folder,Ref_file_type));

Imcounter_R = Imstart_R;
fR2 = struct2cell(fR);
filesR = sort_nat(fR2(1,:));
i = 1;

while((Imcounter_R+1) <= finalImage_R)
    filenamesR{i} = filesR(Imcounter_R+1);
    Imcounter_R = Imcounter_R + Frame_skip;
    i=i+1;
end

totalImagesR = length(filenamesR);

if skip == 0
    for i = 1:totalImagesR
        Im = imread(char(fullfile(fR(1).folder,filenamesR{i})));
        sz = size(Im);
        Im = Im((Initial_crop(1):(sz(1)-Initial_crop(2))),Initial_crop(3):(sz(2)-Initial_crop(4)),:);
        Im1 = imresize(Im,CNN_resize,'lanczos3');  

        template1 = CNN_Create_Template_Filter(net,Im1,actLayer,finalMaps);
        %template1 = CNN_Create_Template(net,Im1,actLayer);
        %Now store in template matrix:
        Template_count = Template_count + 1;
        Template_plot(Template_count,1) = i;
        Template_plot(Template_count,2) = Template_count;
        Template_array(Template_count,:) = template1;
    end
end
Template_count_for_plot = Template_count;
O = zeros(totalImagesR,1);

Query_file_type = strcat('*',Query_file_type);
fQ = dir(fullfile(Query_folder,Query_file_type));

Imcounter_Q = Imstart_Q;
fQ2 = struct2cell(fQ);
filesQ = sort_nat(fQ2(1,:));
i = 1;

while((Imcounter_Q+1) <= finalImage_Q)
    filenamesQ{i} = filesQ(Imcounter_Q+1);
    Imcounter_Q = Imcounter_Q + Frame_skip;
    i=i+1;
end

totalImagesQ = length(filenamesQ);
time = zeros(1,totalImagesQ);

for i = 1:totalImagesQ
    tic
    Im = imread(char(fullfile(fQ(1).folder,filenamesQ{i})));
    
    sz = size(Im);
    Im = Im(Initial_crop(1):(sz(1)-Initial_crop(2)),Initial_crop(3):(sz(2)-Initial_crop(4)),:);

    Im1 = imresize(Im,CNN_resize,'lanczos3');    %for CNN

    sum_array = CNN_Create_Template_Filter(net,Im1,actLayer,finalMaps);

    diffVector = pdist2(sum_array,Template_array,'cosine');

    mx1 = max(diffVector); 
    df1 = mx1 - min(diffVector); 
    %normalise such that 0.999 is best and 0.001 is worst
    for k = 1:Template_count
        O_diff = ((mx1 - diffVector(k))/df1)-epsilon;  %normalise to range 0.001 to 0.999    
        if O_diff < obsThresh %only used to ensure min is not -0.001
            O(k) = epsilon; 
        else
            O(k) = O_diff;
        end 
    end
    
    [minval,id] = max(O);
    window = max(1, id-Rwindow):min(length(O), id+Rwindow);
    not_window = setxor(1:length(O), window);
    min_value_2nd = max(O(not_window));
    quality = log(minval) / log(min_value_2nd);
    
    %loop through every threshold to generate the PR curve.
    for thresh_counter = 1:length(thresh)
        if Dataset == 1
            if quality > thresh(thresh_counter) 
                if sum(GT_file.GPSMatrix(:,Imstart_Q+i))==0
                    %true negative
                else
                    %false negative
                    false_negative_count(thresh_counter) = false_negative_count(thresh_counter) + 1;
                end
            else  % row, column: y, x
                if (GT_file.GPSMatrix(Imstart_R+((Frame_skip*id)-2),Imstart_Q+((Frame_skip*i)-2))==1)
                    %true positive
                    recall_count(thresh_counter) = recall_count(thresh_counter) + 1;
                else  %false positive
                    error_count(thresh_counter) = error_count(thresh_counter) + 1;
                end
            end
        else
            if quality > thresh(thresh_counter) 
                false_negative_count(thresh_counter) = false_negative_count(thresh_counter) + 1;     
            else
                if (i > (id-11)) && (i < (id+11))  %up to 10 frames out in either direction
                    recall_count(thresh_counter) = recall_count(thresh_counter) + 1;
                else
                    error_count(thresh_counter) = error_count(thresh_counter) + 1;
                end
            end
        end
    end
    if plot_skip == 0
        ImCompare = imread(char(fullfile(fR(1).folder,filenamesR{id})));
        subplot(2,2,1,'replace');
        image(Im);
        title('Current View');
        subplot(2,2,2,'replace');
        image(ImCompare)
        title('Matched Scene');
        drawnow;
    end
    time(i) = toc;
end
AvTime = mean(time);
TotalTime = sum(time);
for thresh_counter = 1:length(thresh)
    %Recall = true positives / (true positives + false negatives)
    recall(thresh_counter) = recall_count(thresh_counter)/(recall_count(thresh_counter) + false_negative_count(thresh_counter));
    %Precision = true positives / (true positives + false positives)
    precision(thresh_counter) = recall_count(thresh_counter)/(recall_count(thresh_counter) + error_count(thresh_counter));
end
