close all; clear; clc;

%% === < Removing repeated labels > ===
file_dir = dir('./stage_2_train_images/*.dcm');
label_raw = readtable('stage_2_train_labels.csv');
label_ID = label_raw(:,[1,6]);
uni_label = unique(label_ID,'rows');
writetable(uni_label,'stage_2_train_uni_labels.csv')

%% === < Data analysis > ===
n_Pneumonia = sum(uni_label.Target);
n_Control = length(uni_label.Target) - n_Pneumonia;
prob_Pneumonia = n_Pneumonia / length(uni_label.patientId);

% figure
% histogram(uni_label.Target)

%% === < Splitting files with labels > ===
label_Pneumonia = uni_label(uni_label.Target==1,:);
label_Control = uni_label(uni_label.Target==0,:);

%% === < Copying data > ===
old_folder = './stage_2_train_images';

new_folder_Pneumonia = './train_images/raw/Pneumonia';
% mkdir(new_folder_Pneumonia)
fprintf('Total: %d, File: ',n_Pneumonia)
for idx = 1:n_Pneumonia
    if idx > 1
        fprintf('\b\b\b\b')
    end
    fprintf('%04d',idx)
    old_filename = [char(label_Pneumonia.patientId(idx)),'.dcm'];
    new_filename = old_filename;
    old_path = fullfile(old_folder,old_filename);
    new_path = fullfile(new_folder_Pneumonia,new_filename);
%     copyfile(old_path,new_path);
end
fprintf('\nFinish!!!\n')
 
new_folder_Control = './train_images/raw/Control';
% mkdir(new_folder_Control)
fprintf('Total: %d, File: ',n_Control)
for idx = 1:n_Control
    if idx > 1
        fprintf('\b\b\b\b\b')
    end
    fprintf('%05d',idx)
    old_filename = [char(label_Control.patientId(idx)),'.dcm'];
    new_filename = old_filename;
    old_path = fullfile(old_folder,old_filename);
    new_path = fullfile(new_folder_Control,new_filename);
%     copyfile(old_path,new_path);
end
fprintf('\nFinish!!!\n')

%% === < Splitting training, validation, testing data > ===
prob_testing = 0.1;
rng('default')
rng(2021)

all_folder_Pneumonia = new_folder_Pneumonia;
test_folder_Pneumonia = './train_images/testing/Pneumonia';
% mkdir(test_folder_Pneumonia)
model_folder_Pneumonia = './train_images/model_data/Pneumonia';
% mkdir(model_folder_Pneumonia)
% === random sampling 10% data for testing
n_test_Pneumonia = fix(n_Pneumonia*prob_testing);
loc_test_Pneumonia = randsample(n_Pneumonia,n_test_Pneumonia);
label_Pneumonia_test = label_Pneumonia(loc_test_Pneumonia,:);
% writetable(label_Pneumonia_test,'./train_images/testing/testing_Pneumonia_labels.csv')
label_Pneumonia_model = label_Pneumonia;
label_Pneumonia_model(loc_test_Pneumonia,:) = [];
n_model_Pneumonia = length(label_Pneumonia_model.Target);
% writetable(label_Pneumonia_model,'./train_images/model_data/model_data_Pneumonia_labels.csv')
% === loop for copying
fprintf('Total Testing Pneumonia: %d, File: ',n_test_Pneumonia)
for idx = 1:n_test_Pneumonia
    if idx > 1
        fprintf('\b\b\b\b')
    end
    fprintf('%04d',idx)
    all_filename = [char(label_Pneumonia_test.patientId(idx)),'.dcm'];
    test_filename = all_filename;
    all_path = fullfile(all_folder_Pneumonia,all_filename);
    test_path = fullfile(test_folder_Pneumonia,test_filename);
%     copyfile(all_path,test_path);
end
fprintf('\nFinish!!!\n')
% === loop for copying
fprintf('Total Data for Model Pneumonia: %d, File: ',n_model_Pneumonia)
for idx = 1:n_model_Pneumonia
    if idx > 1
        fprintf('\b\b\b\b')
    end
    fprintf('%04d',idx)
    all_filename = [char(label_Pneumonia_model.patientId(idx)),'.dcm'];
    model_filename = all_filename;
    all_path = fullfile(all_folder_Pneumonia,all_filename);
    model_path = fullfile(model_folder_Pneumonia,model_filename);
%     copyfile(all_path,model_path);
end
fprintf('\nFinish!!!\n')

all_folder_Control = new_folder_Control;
test_folder_Control = './train_images/testing/Control';
% mkdir(test_folder_Control)
model_folder_Control = './train_images/model_data/Control';
% mkdir(model_folder_Control)
% === random sampling 10% data for testing
loc_test_Control = randsample(n_Control,n_test_Pneumonia);
label_Control_test = label_Control(loc_test_Control,:);
% writetable(label_Control_test,'./train_images/testing/testing_Control_labels.csv')
label_Control_model = label_Control;
label_Control_model(loc_test_Control,:) = [];
writetable(label_Control_model,'./train_images/model_data/nonTesting_Control_labels.csv')
loc_model_Control = randsample(n_Control-n_test_Pneumonia,n_model_Pneumonia);
label_Control_model_down = label_Control_model(loc_model_Control,:);
% writetable(label_Control_model_down,'./train_images/model_data/model_data_Control_labels.csv')
% === loop for copying
fprintf('Total Testing Control: %d, File: ',length(loc_test_Control))
for idx = 1:length(loc_test_Control)
    if idx > 1
        fprintf('\b\b\b\b\b')
    end
    fprintf('%05d',idx)
    all_filename = [char(label_Control_test.patientId(idx)),'.dcm'];
    test_filename = all_filename;
    all_path = fullfile(all_folder_Control,all_filename);
    test_path = fullfile(test_folder_Control,test_filename);
%     copyfile(all_path,test_path);
end
fprintf('\nFinish!!!\n')
% === loop for copying
fprintf('Total Data for Model Control: %d, File: ',n_model_Pneumonia)
for idx = 1:n_model_Pneumonia
    if idx > 1
        fprintf('\b\b\b\b\b')
    end
    fprintf('%05d',idx)
    all_filename = [char(label_Control_model.patientId(idx)),'.dcm'];
    model_filename = all_filename;
    all_path = fullfile(all_folder_Control,all_filename);
    model_path = fullfile(model_folder_Control,model_filename);
%     copyfile(all_path,model_path);
end
fprintf('\nFinish!!!\n')
