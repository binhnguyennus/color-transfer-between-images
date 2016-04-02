clear all
close all

workspace;	% Make sure the workspace panel is showing.
fontSize = 16;

load('.\reference_colors.mat','reference_colors');
load('.\target_colors.mat','target_colors');
% target_image = im2double(imread('target.jpg'));
target_image = im2double(imread('.\scene\DSC_0257_without_noise_2012_xrite_stageProPhoto.tif'));
figure, imshow(target_image)
load ('data.mat', 'data');
% [data,ps] = removerows(data,'ind',[15]);
% [data,ps] = removerows(data,'ind',[23]);
% 
% reference_colors =[data; reference_colors];
% target_colors = [data; target_colors];

title('Before', 'FontSize', fontSize);
% [T, f] = optTwithMinAE(target_colors, reference_colors);
T = target_colors\reference_colors;
target_image = apply_cmatrix(target_image,T);
target_image = max(0,min(target_image,1));

% [x,y,z]  = size(target_image);
% 
% for i=1:x
%     for j=1:y
%         pixel_value = [target_image(x,y,1) target_image(x,y,2) target_image(x,y,3)];
%         target_image(x,y,:) = pixel_value*T;
%     end
% end 

figure, imshow(target_image)

title('After', 'FontSize', fontSize);
