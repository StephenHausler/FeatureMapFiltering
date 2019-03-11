%--------------------------------------------------------------------------
%   Project Title: Feature Map Filtering
%   Author: Stephen Hausler
%   
%   Open Source Code, requires MATLAB with Neural Network Toolbox.
%   Refer to LICENSES.txt for license to this source code and 3RD_PARTY_
%   LICENSES for all 3rd party licences.
%-------------------------------------------------------------------------

function [pass] = Feature_Map_Filter(actLayer,delta,net)

fmap_scoring = zeros(50,600);
  
Dataset = 1;   %1: St Lucia, 2: Nordland, 3: Oxford RobotCar

%for St Lucia: (replace with your own folders)
folder_Q = 'D:\Windows\St_Lucia_Dataset\1545_15FPS\Frames';
folder_R = 'D:\Windows\St_Lucia_Dataset\0845_15FPS\Frames';
load('D:\Windows\St_Lucia_Dataset\1545_15FPS\fGPS_Lin.mat'); %GPS1545_Lin
load('D:\Windows\St_Lucia_Dataset\0845_15FPS\fGPS_Lin.mat'); %GPS0845_Lin
imstart_Q = 1;
imstart_R = 1;

%for Nordland:
%folder_Q = 'D:\Windows\Nordland\nordland_winter_images';
%folder_R = 'D:\Windows\Nordland\nordland_summer_images';
%sequences are aligned, so find corresponding image.

%for Oxford RobotCar:
%folder_Q = 'D:\Windows\oxford-data\2014-12-10-18-10-50\stereo\left_rect';
%folder_R = 'D:\Windows\oxford-data\2014-12-09-13-21-02\stereo\left_rect';

imageCounter = 0; 

figure

for i = 200:10:690 %50 calibration images %lucia
%for i = 2700:5:2945 %Nordland
    tic
    imageCounter = imageCounter + 1;
    
    Im_Q = imread(char(fullfile(folder_Q,sprintf('I0%d.jpeg',i)))); %Query image
    Im_Q = imresize(Im_Q,[227 227],'lanczos3');  %HybridNet
    %Im_Q = imresize(Im_Q,[224 224],'lanczos3');  %vgg-16
    if Dataset == 1
        %use GT to find corresponding reference image.
        currFrameGPSLat = GPS1545_Lin.fGPS((imstart_Q+i-1),1);
        currFrameGPSLong = GPS1545_Lin.fGPS((imstart_Q+i-1),2);
        min_d = 1e6;
        for id = 1:4000
            guessFrameGPSLong = GPS0845_Lin.fGPS((imstart_R+id-1),2);
            guessFrameGPSLat = GPS0845_Lin.fGPS((imstart_R+id-1),1);
            d = GPS2Meters(currFrameGPSLat,currFrameGPSLong,guessFrameGPSLat,guessFrameGPSLong);
            if d < min_d
                min_d = d;
                d_id = id;
            end
        end
    else
        d_id = i;  %traverses are aligned on Nordland
    end
    Im_R = imread(char(fullfile(folder_R,sprintf('I0%d.jpeg',d_id)))); %Reference image    
    Im_R = imresize(Im_R,[227 227],'lanczos3');  %HybridNet
    %Im_R = imresize(Im_R,[224 224],'lanczos3');  %vgg-16
    
    r = randi([1000 3900],1,1);  %Lucia
    %r = randi([3700 5000],1,1);  %Nordland
    
    Im_R2 = imread(char(fullfile(folder_R,sprintf('I%d.jpeg',r)))); %a random image somewhere else in the reference traverse
    Im_R2 = imresize(Im_R2,[227 227],'lanczos3');  %HybridNet
    %Im_R2 = imresize(Im_R2,[224 224],'lanczos3');  %vgg-16
    
    act_Q = activations(net,Im_Q,actLayer,'OutputAs','channels','ExecutionEnvironment','gpu');
    act_R = activations(net,Im_R,actLayer,'OutputAs','channels','ExecutionEnvironment','gpu');
    act_R2 = activations(net,Im_R2,actLayer,'OutputAs','channels','ExecutionEnvironment','gpu');
    
    sz1 = size(act_Q); 
    act_Q = reshape(act_Q,[sz1(1) sz1(2) 1 sz1(3)]);   
    sh11 = ceil(sz1(1)/2); sh12 = ceil(sz1(2)/2);  
    feats_Q = 0;    
    for j = 1:sz1(3)
        feats_Q(1,j) = max(max(act_Q(:,:,1,j)));
        feats_Q(2,j) = max(max(act_Q(1:sh11,1:sh12,1,j)));
        feats_Q(3,j) = max(max(act_Q(1:sh11,sh12:sz1(2),1,j)));
        feats_Q(4,j) = max(max(act_Q(sh11:sz1(1),1:sh12,1,j)));
        feats_Q(5,j) = max(max(act_Q(sh11:sz1(1),sh12:sz1(2),1,j)));
    end
    
    sz1 = size(act_R); 
    act_R = reshape(act_R,[sz1(1) sz1(2) 1 sz1(3)]);   
    sh11 = ceil(sz1(1)/2); sh12 = ceil(sz1(2)/2);  
    feats_R = 0;    
    for j = 1:sz1(3)
        feats_R(1,j) = max(max(act_R(:,:,1,j)));
        feats_R(2,j) = max(max(act_R(1:sh11,1:sh12,1,j)));
        feats_R(3,j) = max(max(act_R(1:sh11,sh12:sz1(2),1,j)));
        feats_R(4,j) = max(max(act_R(sh11:sz1(1),1:sh12,1,j)));
        feats_R(5,j) = max(max(act_R(sh11:sz1(1),sh12:sz1(2),1,j)));
    end
    
    sz1 = size(act_R2);
    act_R2 = reshape(act_R2,[sz1(1) sz1(2) 1 sz1(3)]);
    sh11 = ceil(sz1(1)/2); sh12 = ceil(sz1(2)/2);
    feats_R2 = 0;
    for j = 1:sz1(3)
        feats_R2(1,j) = max(max(act_R2(:,:,1,j)));
        feats_R2(2,j) = max(max(act_R2(1:sh11,1:sh12,1,j)));
        feats_R2(3,j) = max(max(act_R2(1:sh11,sh12:sz1(2),1,j)));
        feats_R2(4,j) = max(max(act_R2(sh11:sz1(1),1:sh12,1,j)));
        feats_R2(5,j) = max(max(act_R2(sh11:sz1(1),sh12:sz1(2),1,j)));
    end
    
    sz1 = size(feats_R);
    not_Min = 0;
    prev_maxval = -1e6;
    fCounter = 0;
    numFeats = sz1(2);
    totalFeats = sz1(2);
    clear fmap_track
    clear distSameStore
    clear distDiffStore
    
    %hybrid net produces 256 feature maps from Conv-5.
    while not_Min == 0      %Greedy search algorithm
        fCounter = fCounter + 1;
        for j = 1:numFeats
            if j == 1
                vector_Q = reshape(feats_Q(:,2:length(feats_Q))...
                    ,[1 sz1(1)*(sz1(2)-1)]);
                vector_R = reshape(feats_R(:,2:length(feats_R))...
                    ,[1 sz1(1)*(sz1(2)-1)]);
                vector_R2 = reshape(feats_R2(:,2:length(feats_R2))...
                    ,[1 sz1(1)*(sz1(2)-1)]);
            elseif j == numFeats
                vector_Q = reshape(feats_Q(:,1:(length(feats_Q)-1))...
                    ,[1 sz1(1)*(sz1(2)-1)]);
                vector_R = reshape(feats_R(:,1:(length(feats_R)-1))...
                    ,[1 sz1(1)*(sz1(2)-1)]);
                vector_R2 = reshape(feats_R2(:,1:(length(feats_R)-1))...
                    ,[1 sz1(1)*(sz1(2)-1)]);
            else
                %now gotta splice the two halfs of feats_Q & R together...
                feats_Q_section1 = feats_Q(:,1:(j-1));
                feats_Q_section2 = feats_Q(:,(j+1):length(feats_Q));
                feats_R_section1 = feats_R(:,1:(j-1));
                feats_R_section2 = feats_R(:,(j+1):length(feats_R));
                feats_R2_section1 = feats_R2(:,1:(j-1));
                feats_R2_section2 = feats_R2(:,(j+1):length(feats_R2));
                
                vector_Q = reshape([feats_Q_section1 feats_Q_section2],[1 sz1(1)*(sz1(2)-1)]);
                vector_R = reshape([feats_R_section1 feats_R_section2],[1 sz1(1)*(sz1(2)-1)]);
                vector_R2 = reshape([feats_R2_section1 feats_R2_section2],[1 sz1(1)*(sz1(2)-1)]);
            end
            %now calc L2 distance between vector_Q and vector_R.
            distSame(j) = pdist2(vector_Q,vector_R,'euclidean'); %same location, different time of day
            distDiff(j) = pdist2(vector_R,vector_R2,'euclidean');  %different location, same time of day
        end
        distSameStore(fCounter,:) = padarray(distSame,[0 (totalFeats-numFeats)],-1,'post');
        distDiffStore(fCounter,:) = padarray(distDiff,[0 (totalFeats-numFeats)],-1,'post');
        diff = distDiff - distSame;
        %shouldnt use abs here
        %because want distDiff to be greater than distSame!
        
        [maxval,worst_Fmap] = max(diff); %which worst feature map to remove results in
        %the greatest distance between the same location and a different
        %location while minimising the distance between the same location at 
        %different times of day.
        clear distSame
        clear distDiff
        clear diff
        %check this against previous score (need to find global min cosine
        %distance)
        
        %remove worst Fmap from both featsR and featsQ and re-splice the
        %vectors back together again...
        if worst_Fmap == 1
            feats_Q = feats_Q(:,2:length(feats_Q));
            feats_R = feats_R(:,2:length(feats_R));
            feats_R2 = feats_R2(:,2:length(feats_R2));
        elseif worst_Fmap == numFeats
            feats_Q = feats_Q(:,1:(length(feats_Q)-1));
            feats_R = feats_R(:,1:(length(feats_R)-1));
            feats_R2 = feats_R2(:,1:(length(feats_R2)-1));
        else    
            feats_Q_section1 = feats_Q(:,1:(worst_Fmap-1));
            feats_Q_section2 = feats_Q(:,(worst_Fmap+1):length(feats_Q));
            feats_R_section1 = feats_R(:,1:(worst_Fmap-1));
            feats_R_section2 = feats_R(:,(worst_Fmap+1):length(feats_R));
            feats_R2_section1 = feats_R2(:,1:(worst_Fmap-1));
            feats_R2_section2 = feats_R2(:,(worst_Fmap+1):length(feats_R2));
            
            feats_Q = [feats_Q_section1 feats_Q_section2];
            feats_R = [feats_R_section1 feats_R_section2];
            feats_R2 = [feats_R2_section1 feats_R2_section2];
        end
        fmap_track(fCounter) = worst_Fmap;
        numFeats = numFeats - 1;
        sz1(2) = sz1(2) - 1;
        
        if maxval > (prev_maxval + delta) %delta: minimum gradient for early stopping
            prev_maxval = maxval;
        else
            not_Min = 1;
        end
    end
    %convert fmap_track to true, original, fmap positions:
    trueFmapPos = 1:totalFeats;
    for k = 1:fCounter
        fmap_scoring(imageCounter,trueFmapPos(fmap_track(k))) = 1;
        if fmap_track(k) == 1
            trueFmapPos = trueFmapPos(2:length(trueFmapPos));
        elseif fmap_track(k) == length(trueFmapPos)
            trueFmapPos = trueFmapPos(1:(length(trueFmapPos)-1));
        else
            trueFmapPos_section1 = trueFmapPos(1:(fmap_track(k)-1));
            trueFmapPos_section2 = trueFmapPos((fmap_track(k)+1):length(trueFmapPos));
            trueFmapPos = [trueFmapPos_section1 trueFmapPos_section2];
        end
    end     
    min_cosines(imageCounter) = maxval;
    track_final_map_counts(imageCounter) = numFeats;
    toc
end

fmap_scoring = fmap_scoring(:,1:totalFeats);
finalScores(1,:) = 1:totalFeats;
finalScores(2,:) = sum(fmap_scoring,1);

max_map_count = max(track_final_map_counts);
%find the 'max_map_count' best feature maps

saveFinalScores = finalScores;

for k = 1:max_map_count
    [~,i] = min(finalScores(2,:));
    finalMaps(1,k) = i;
    finalScores(2,i) = 10000;
end


finalScores = saveFinalScores;

save(sprintf('StLucia_finalMaps_HybridNet_earlyStop_0%.0f_actLayer_%d.mat',...
    (10*delta),actLayer),'finalMaps','max_map_count','finalScores');

%then run a VPR algorithm using finalMaps

pass =1;




