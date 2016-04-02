clear all
close all

workspace;	% Make sure the workspace panel is showing.
fontSize = 16;

load('.\reference_colors.mat','reference_colors');
load('.\target_colors.mat','target_colors');
% target_image = im2double(imread('target.jpg'));
target_image = im2double(imread('IMG_0472.tif'));
figure, imshow(target_image)

title('Before', 'FontSize', fontSize);
[T, f] = optTwithMinAE(target_colors, reference_colors);

target_image = apply_cmatrix(target_image,T);
% target_image = max(0,min(target_image,1));


figure, imshow(target_image)

title('After', 'FontSize', fontSize);
