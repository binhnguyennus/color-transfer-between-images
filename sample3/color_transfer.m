close all
clear all


reference = im2double(imread('DSC_0247_without_noise_2012_xrite_stageProPhoto.tif'));
target= im2double(imread('.\scene\DSC_0257_without_noise_2012_xrite_stageProPhoto.tif'));
workspace;	% Make sure the workspace panel is showing.
fontSize = 16;



reference_colors = zeros(0,3);
target_colors = zeros(0,3);

i=1;
while(1)
    figure, imshow(reference)

    title('Reference', 'FontSize', fontSize);

    [x,y,z] = size(reference);
    mask1 = zeros(x, y);    
    fprintf('Patch = %f\n', i);  % Method 1
    h = imrect;
    position = wait(h);
    mask1(position(2):position(2)+position(4), position(1):position(1)+position(3)) = 1;
%     figure, imshow(mask);
    close all

    figure, imshow(target)
    title('Target', 'FontSize', fontSize);

    [x2,y2,z2] = size(target);
    mask2 = zeros(x2, y2);    
    fprintf('Patch = %f\n', i);  % Method 1
    h = imrect;
    position = wait(h);
    mask2(position(2):position(2)+position(4), position(1):position(1)+position(3)) = 1;
    
    ref_r = reference(:,:,1) .* mask1;
    ref_g = reference(:,:,2) .* mask1;
    ref_b = reference(:,:,3) .* mask1;
    
    tar_r = target(:,:,1) .* mask2;
    tar_g = target(:,:,2) .* mask2;
    tar_b = target(:,:,3) .* mask2;
        
    ref_patch = [mean(nonzeros(ref_r)); mean(nonzeros(ref_g)); mean(nonzeros(ref_b))];
    target_patch= [mean(nonzeros(tar_r)); mean(nonzeros(tar_g)); mean(nonzeros(tar_b))];
    reference_colors = [reference_colors; ref_patch'];
    target_colors = [target_colors; target_patch'];
    close all
    i=i+1;
    prompt = 'Do you want more? Y/N [Y]: ';
    str = input(prompt,'s');
    if isempty(str)
        str = 'Y';
    end
    if str == 'N'
        break;
    end
end 
% load ('data.mat', 'data');
% [data,ps] = removerows(data,'ind',[11]);
% [data,ps] = removerows(data,'ind',[6]);
% reference_colors =[data; reference_colors];
% target_colors = [data; target_colors];
target_colors = target_colors.*255;
reference_colors = reference_colors.*255;
save('reference_colors.mat','reference_colors');
save('target_colors.mat','target_colors');







