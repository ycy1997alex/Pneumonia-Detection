close all; clear; clc;
set(0,'defaultAxesFontSize',13)

%% === < image importing > ===
img_path = 'Data/testing_fig/Pneumonia/fffb2395-8edd-4954-8a89-ffe2fd329be3.png';
img = imread(img_path);

%% === < model checking > ===
mat_dir = dir('Info/*.mat');

%% === < model importing and ensembling > ===
PredProb = zeros(1,2);
pred_e = [];
for idx_model = 1:length(mat_dir)
    load(fullfile(mat_dir(idx_model).folder,mat_dir(idx_model).name))
    img = imresize(img,net.Layers(1).InputSize(1:2));
    img_3d = uint8(zeros(net.Layers(1).InputSize));
    for idx_channel = 1:3
        img_3d(:,:,idx_channel) = img;
    end
    % === probability
    TestPredProb = predict(net,img_3d);
    % === classification
    TestPred = classify(net,img_3d);
    % === Ensemble
    PredProb = PredProb + (1/length(mat_dir))*TestPredProb;
    pred_e = [pred_e;TestPred];
end
[prob,loc] = max(PredProb);
% Control / Pneumonia
if loc == 2
    Pred = 'Pneumonia';
elseif loc == 1
    Pred = 'Control';
else
    error('No this Type')
end

%%
modelname = 'Ensemble';
if contains(modelname,'Ensemble')
    fprintf('Yes\n')
else
    fprintf('No\n')
end