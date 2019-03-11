%--------------------------------------------------------------------------
%   Project Title: Feature Map Filtering
%   Author: Stephen Hausler
%   
%   Open Source Code, requires MATLAB with Neural Network Toolbox.
%   Refer to LICENSES.txt for license to this source code and 3RD_PARTY_
%   LICENSES for all 3rd party licences.
%-------------------------------------------------------------------------

function [template] = CNN_Create_Template(net,Im,actLayer1)

act = activations(net, Im, actLayer1,'OutputAs','channels','ExecutionEnvironment','gpu');

sz1 = size(act); 

act1 = reshape(act,[sz1(1) sz1(2) 1 sz1(3)]);

sh11 = ceil(sz1(1)/2); sh12 = ceil(sz1(2)/2);

sum_array=0;

for j = 1:sz1(3)
    sum_array(1,j) = max(max(act1(:,:,1,j)));
    sum_array(2,j) = max(max(act1(1:sh11,1:sh12,1,j)));
    sum_array(3,j) = max(max(act1(1:sh11,sh12:sz1(2),1,j)));        
    sum_array(4,j) = max(max(act1(sh11:sz1(1),1:sh12,1,j)));
    sum_array(5,j) = max(max(act1(sh11:sz1(1),sh12:sz1(2),1,j)));
end

sz1 = size(sum_array);
    
template = reshape(sum_array,[1 sz1(1)*sz1(2)]);

end




