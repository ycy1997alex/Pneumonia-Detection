% DarkNet53
close all; clear; clc;
timenow = datestr(now,'yyyymmdd_HHMMSS');
fprintf('Time: %s\n',timenow)
modelName = 'DarkNet-53';
outputFolder = 'Result';
if not(isfolder(outputFolder))
    mkdir(outputFolder)
end
set(0,'defaultAxesFontSize',13)

%% === < data importing > ===
% === data source folder
data_folder = 'Data/model_data_fig';
% === image data store to loading images
imds = imageDatastore(data_folder, ...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames',...
    'ReadFcn',@resizeTF); % using ReadFcn to reading specific image format

%% === < checking label and data balance > ===
labelCountTable = countEachLabel(imds);
labelCount = labelCountTable.Count;
min_labelCount = min(labelCount);

%% === < defining training and validation ratio > ===
% === setting ratio of training data
train_ratio = 0.7;
% === getting number of training data
numTrainFiles = fix(min_labelCount*train_ratio);
% === spliting training and validation data
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
% === getting number of category
numClasses = numel(categories(imdsTrain.Labels));

%% === < showing some figures > ===
% numTrainImages = numel(imdsTrain.Labels);
% idx = randperm(numTrainImages,16);
% figure
% for i = 1:16
%     subplot(4,4,i)
%     I = readimage(imdsTrain,idx(i));
%     imshow(I)
% end

%% === < transfer learning model > ===
TFmodel = darknet53;
% analyzeNetwork(TFmodel)

%% === < checking image size > ===
% === reading images and getting size of them
img = readimage(imds,1);
imgSize = size(img);
% === getting size of transfer learning model
inputSize = TFmodel.Layers(1).InputSize;
% === checking image size for input images and the image input layer of the model
if imgSize == inputSize
    fprintf('Image size is %d x %d x %d.\n',imgSize(1),imgSize(2),imgSize(3))
else
    error('Image size does not match.')
end

%% === < defining layers > ===
% === defining model
lgraph = layerGraph(TFmodel);
% === removing the unwanted layers
lgraph = removeLayers(lgraph,TFmodel.Layers(end).Name);
lgraph = removeLayers(lgraph,TFmodel.Layers(end-1).Name);
lgraph = removeLayers(lgraph,TFmodel.Layers(end-2).Name);
% === adding and connecting the needed layers
lgraph = addLayers(lgraph,fullyConnectedLayer(numClasses,'Name','fc_numClasses', ...
    'WeightLearnRateFactor',5,'BiasLearnRateFactor',5));
lgraph = addLayers(lgraph,softmaxLayer('Name','softmax'));
lgraph = addLayers(lgraph,classificationLayer('Name','classOutput'));
lgraph = connectLayers(lgraph,TFmodel.Layers(end-3).Name,'fc_numClasses');
lgraph = connectLayers(lgraph,'fc_numClasses','softmax');
lgraph = connectLayers(lgraph,'softmax','classOutput');
% analyzeNetwork(lgraph)

%% === < defining options to train a model > ===
% === setting mini-batch size for each iteration (could be any positive integer)
miniBatchSize = 16;
% === setting validation frequency to validate (could be any positive integer)
validationFrequency = floor(numTrainFiles/miniBatchSize);
% === setting validation patience for early stopping (could be any positive integer)
validationPatience = 10;
% === training options
% --- setting 'LearnRateSchedule' as piecewise to changing learning rate
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',validationFrequency, ...
    'ValidationPatience',validationPatience, ...
    'LearnRateSchedule','piecewise', ...
    'Verbose',true, ...
    'Plots','training-progress', ... % {'CheckpointPath',''} could be added
    'ExecutionEnvironment','auto'); % 'ExecutionEnvironment': 'auto','cpu','gpu'

%% === < image data augmentation > ===
% === image data augmentation setting
augmenter = imageDataAugmenter( ...
    'RandRotation',[-10 10], ...
    'RandScale',[0.9 1.1], ...
    'RandXTranslation',[-20 20], ...
    'RandYTranslation',[-20 20], ...
    'RandXReflection',true);
% === augmented image datastore
augimds = augmentedImageDatastore(inputSize,imdsTrain,'DataAugmentation',augmenter);

%% === < training network > ===
[net,info] = trainNetwork(augimds,lgraph,options);

%% === < training progress > ===
fig = figure;
fig.Position(1) = 10;
fig.Position(2) = 50;
fig.Position(3) = 1500;
fig.Position(4) = 700;
tiledlayout(2,1,'TileSpacing','tight','Padding','tight')
% --- accuracy
nexttile
plot(info.TrainingAccuracy,'-','Color','#0072BD','DisplayName','Training')
hold on
plot(info.ValidationAccuracy,'ko','LineWidth',2,'DisplayName','Validation')
hold on
text_acc = sprintf('Final: %.2f',info.FinalValidationAccuracy);
scatter(length(info.ValidationAccuracy),info.ValidationAccuracy(end),'ko','filled','DisplayName',text_acc)
title(sprintf('Training Progress (%s)',modelName))
ylabel('Accuracy (%)')
ylim([0 100])
grid on
xticklabels([])
lgd = legend('Location','northeastoutside');
title(lgd,'Accuracy')
% --- loss
nexttile
plot(info.TrainingLoss,'-','Color','#D95319','DisplayName','Training')
hold on
plot(info.ValidationLoss,'kd','LineWidth',2,'DisplayName','Validation')
hold on
text_loss = sprintf('Final: %.3f',info.FinalValidationLoss);
scatter(length(info.ValidationLoss),info.ValidationLoss(end),'kd','filled','DisplayName',text_loss)
xlabel('Iteration')
ylabel('Loss (Cross-Entropy)')
ylim([0 max(max(info.ValidationLoss),max(info.TrainingLoss))+0.2])
grid on
lgd = legend('Location','northeastoutside');
title(lgd,'Loss')
% --- output
figureName = sprintf('%s_%s_TrainingProgress.png',modelName,timenow);
saveas(gcf,fullfile(outputFolder,figureName));

%% === < training result > ===
% === classification
ModelPred = classify(net,imdsTrain);
% === ground truth
ModelLabel = imdsTrain.Labels;
% === predicted probability
ModelPredProb = predict(net,imdsTrain);
% === accuracy
Model_accuracy = sum(ModelPred == ModelLabel)/numel(ModelLabel);
fprintf('Training Accuracy: %.2f%%\n',100*Model_accuracy)
% === confusion matrix
figure
plotconfusion(ModelLabel,ModelPred)
title('Training Confusion Matrix')
figureName = sprintf('%s_%s_Training_ConfusionMatrix.png',modelName,timenow);
saveas(gcf,fullfile(outputFolder,figureName))

%% === < validation result > ===
% === classification
YPred = classify(net,imdsValidation);
% === ground truth
YValidation = imdsValidation.Labels;
% === predicted probability
YPredProb = predict(net,imdsValidation);
% === accuracy
accuracy = sum(YPred == YValidation)/numel(YValidation);
fprintf('Validation Accuracy: %.2f%%\n',100*accuracy)
% === confusion matrix
figure
plotconfusion(YValidation,YPred)
title('Validation Confusion Matrix')
figureName = sprintf('%s_%s_Validation_ConfusionMatrix.png',modelName,timenow);
saveas(gcf,fullfile(outputFolder,figureName))

%% === < testing result > ===
% === data source folder
test_data_src = 'Data/testing_fig/';
% === image data store to loading images
imdsTest = imageDatastore(test_data_src, ...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames',...
    'ReadFcn',@resizeTF);
% === classification
TestPred = classify(net,imdsTest);
% === ground truth
TestLabel = imdsTest.Labels;
% === predicted probability
TestPredProb = predict(net,imdsTest);
% === accuracy
accuracy = sum(TestPred == TestLabel)/numel(TestLabel);
fprintf('Testing Accuracy: %.2f%%\n',100*accuracy)
% === confusion matrix
figure
plotconfusion(TestLabel,TestPred)
title('Testing Confusion Matrix')
figureName = sprintf('%s_%s_Testing_ConfusionMatrix.png',modelName,timenow);
saveas(gcf,fullfile(outputFolder,figureName))

%% === < ROC and AUC > ===
category = {'Control','Pneumonia'};

figure
for idx = 1:length(category)
    [X,Y,T,AUC] = perfcurve(cellstr(ModelLabel),ModelPredProb(:,idx),category{idx});
    displayname = sprintf('%s - AUC: %.4f',category{idx},AUC);
    plot(X,Y,'DisplayName',displayname)
    hold on
end
legend('Location','SouthEast')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification (Training)')
figureName = sprintf('%s_%s_Training_ROC.png',modelName,timenow);
saveas(gcf,fullfile(outputFolder,figureName))

figure
for idx = 1:length(category)
    [X,Y,T,AUC] = perfcurve(cellstr(YValidation),YPredProb(:,idx),category{idx});
    displayname = sprintf('%s - AUC: %.4f',category{idx},AUC);
    plot(X,Y,'DisplayName',displayname)
    hold on
end
legend('Location','SouthEast')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification (Validation)')
figureName = sprintf('%s_%s_Validation_ROC.png',modelName,timenow);
saveas(gcf,fullfile(outputFolder,figureName))

figure
for idx = 1:length(category)
    [X,Y,T,AUC] = perfcurve(cellstr(TestLabel),TestPredProb(:,idx),category{idx});
    displayname = sprintf('%s - AUC: %.4f',category{idx},AUC);
    plot(X,Y,'DisplayName',displayname)
    hold on
end
legend('Location','SouthEast')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification (Testing)')
figureName = sprintf('%s_%s_Testing_ROC.png',modelName,timenow);
saveas(gcf,fullfile(outputFolder,figureName))

%% === < output > ===
close all;
filename = sprintf('%s_%s',modelName,timenow);
save(fullfile(outputFolder,filename))
netname = sprintf('net_%s_%s.mat',modelName,timenow);
save(fullfile(outputFolder,netname),'net')

disp('Finish!!!')

%%
function output = resizeTF(filename)

size = 256;
img = imread(filename);
img = imresize(img,[size size]);
img_3d = uint8(zeros([size,size,3]));
for idx = 1:3
    img_3d(:,:,idx) = img;
end
output = img_3d;

end