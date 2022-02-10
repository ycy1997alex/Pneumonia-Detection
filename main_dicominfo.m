close all; clear; clc;

info = dicominfo('Data/model_data/Control/0a0f6755-610d-4b7c-a460-5f5a8f5c0743.dcm');
dcm = dicomread(info);

figure
imshow(dcm,[])

img_name = sprintf('%s.png',info.Filename);
%imwrite(img_new,img_name)