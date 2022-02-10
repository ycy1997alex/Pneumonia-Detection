close all; clear; clc;

%% === < data for model: Control > ===
file_dir = dir('Data/model_data/Control/*.dcm');
new_folder = 'Data/model_data_fig/Control';
if not(isfolder(new_folder))
    mkdir(new_folder)
end
fprintf('Total: %d, File: ',length(file_dir))
parfor idx = 1:length(file_dir)
%     if idx > 1
%         fprintf('\b\b\b\b')
%     end
%     fprintf('%04d',idx)
    dcm = dicomread(fullfile(file_dir(idx).folder,file_dir(idx).name));
    new_figname = sprintf('%s.png',file_dir(idx).name(1:end-4));
    imwrite(dcm,fullfile(new_folder,new_figname))
end
fprintf('\nFinish!!!\n')

%% === < data for model: Pneumonia > ===
file_dir = dir('Data/model_data/Pneumonia/*.dcm');
new_folder = 'Data/model_data_fig/Pneumonia';
if not(isfolder(new_folder))
    mkdir(new_folder)
end
fprintf('Total: %d, File: ',length(file_dir))
parfor idx = 1:length(file_dir)
%     if idx > 1
%         fprintf('\b\b\b\b')
%     end
%     fprintf('%04d',idx)
    dcm = dicomread(fullfile(file_dir(idx).folder,file_dir(idx).name));
    new_figname = sprintf('%s.png',file_dir(idx).name(1:end-4));
    imwrite(dcm,fullfile(new_folder,new_figname))
end
fprintf('\nFinish!!!\n')

%% === < data for testing: Control > ===
file_dir = dir('Data/testing/Control/*.dcm');
new_folder = 'Data/testing_fig/Control';
if not(isfolder(new_folder))
    mkdir(new_folder)
end
fprintf('Total: %d, File: ',length(file_dir))
parfor idx = 1:length(file_dir)
%     if idx > 1
%         fprintf('\b\b\b\b')
%     end
%     fprintf('%04d',idx)
    dcm = dicomread(fullfile(file_dir(idx).folder,file_dir(idx).name));
    new_figname = sprintf('%s.png',file_dir(idx).name(1:end-4));
    imwrite(dcm,fullfile(new_folder,new_figname))
end
fprintf('\nFinish!!!\n')

%% === < data for testing Pneumonia > ===
file_dir = dir('Data/testing/Pneumonia/*.dcm');
new_folder = 'Data/testing_fig/Pneumonia';
if not(isfolder(new_folder))
    mkdir(new_folder)
end
fprintf('Total: %d, File: ',length(file_dir))
parfor idx = 1:length(file_dir)
%     if idx > 1
%         fprintf('\b\b\b\b')
%     end
%     fprintf('%04d',idx)
    dcm = dicomread(fullfile(file_dir(idx).folder,file_dir(idx).name));
    new_figname = sprintf('%s.png',file_dir(idx).name(1:end-4));
    imwrite(dcm,fullfile(new_folder,new_figname))
end
fprintf('\nFinish!!!\n')