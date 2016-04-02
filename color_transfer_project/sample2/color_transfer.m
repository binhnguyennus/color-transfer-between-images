close all
clear all


target = im2double(imread('IMG_0472.tif'));
reference = im2double(imread('IMG_0470.tif'));
workspace;	% Make sure the workspace panel is showing.
fontSize = 16;

prompt = 'What is the number of colors? ';
numberOfColors = input(prompt);

reference_colors = zeros(numberOfColors,3);
target_colors = zeros(numberOfColors,3);


for i=1:numberOfColors
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
    reference_colors(i,:) = ref_patch;   
    target_colors(i,:) = target_patch;   
    close all

end 

save('reference_colors.mat','reference_colors');
save('target_colors.mat','target_colors');







