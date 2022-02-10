close all; clear; clc;
demo_output_folder = 'Demo';
mkdir(demo_output_folder)

%% === < loading trained model > ===
load('Info/net_ResNet-18.mat')

%% === < importing image and classifying > ===
img_path = 'Data/testing_fig/Pneumonia/fffb2395-8edd-4954-8a89-ffe2fd329be3.png';
img = imread(img_path);
img = imresize(img,net.Layers(1).InputSize(1:2));

img_3d = uint8(zeros(net.Layers(1).InputSize));
for idx = 1:3
    img_3d(:,:,idx) = img;
end

[classfn,score] = classify(net,img_3d);

%% === < GradCam and mask> ===
map = gradCAM(net,img_3d,classfn);

sort_mat = sort(map(:), 'descend');
cut_prob = 0.1;
cut_pt = fix(cut_prob*length(sort_mat));
mask = map > sort_mat(cut_pt);
img_roi = uint8(mask).*img_3d;

%% === < image > ===
figure
imshow(img);
title(sprintf("%s (%.4f)", classfn, score(classfn)));
% figureName = 'windowed_img.png';
% saveas(gcf,fullfile(demo_output_folder,figureName))

%% === < image with GradCam heatmap > ===
figure
imagesc(map,'AlphaData',0.1)
colorbar

figure
imagesc(uint8(255*map),'AlphaData',0.1)
colorbar

figure
imshow(img_3d)
hold on
imagesc(map,'AlphaData',0.2);
colormap('jet')
hold off
title("Grad-CAM")
colorbar
% figureName = 'GradCAM_img.png';
% saveas(gcf,fullfile(demo_output_folder,figureName))

%% === < image with mask > ===
figure
imshow(mask)
% figureName = 'mask_top10per.png';
% saveas(gcf,fullfile(demo_output_folder,figureName))

figure
imshow(img_roi)
figureName = 'ROI_dicom.png';
% saveas(gcf,fullfile(demo_output_folder,figureName))
