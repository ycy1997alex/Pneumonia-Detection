close all; clear; clc;
set(0,'defaultAxesFontSize',13)

%% === < model checking > ===
mat_dir = dir('Result/*.mat');

%% === < model importing > ===
load(fullfile(mat_dir(1).folder,mat_dir(1).name))
PredProb = zeros(size(TestPredProb));
for idx = 1:length(mat_dir)
    load(fullfile(mat_dir(idx).folder,mat_dir(idx).name))
    PredProb = PredProb + (1/length(mat_dir))*TestPredProb;
end

%%
[PredProb_value,PredProb_loc] = max(PredProb,[],2);
PredProb_type_list = {};
% Control / Pneumonia
for idx = 1:length(PredProb_loc)
    PredProb_loc_idx = PredProb_loc(idx);
    if PredProb_loc_idx == 1
        PredProb_type = 'Control';
    elseif PredProb_loc_idx == 2
        PredProb_type = 'Pneumonia';
    else
        error('Wrong!!!');
    end
    PredProb_type_list{idx,1} = PredProb_type;
end

Pred = categorical(PredProb_type_list);

%% === < MCS testing result > ===
outputFolder = 'Result/Ensemble/';
mkdir(outputFolder)
% === ground truth
Label = TestLabel;
% === accuracy
accuracy = sum(Pred == Label)/numel(Label);
fprintf('Testing Accuracy: %.2f%%\n',100*accuracy)
% === confusion matrix
figure
plotconfusion(Label,Pred)
title('Testing Confusion Matrix (Ensemble)')
% figureName = sprintf('Ensemble_Testing_ConfusionMatrix.png');
% saveas(gcf,fullfile(outputFolder,figureName))
% === evaluation
confusionMat = confusionmat(Label,Pred);
precision = @(confusionMat) diag(confusionMat)./sum(confusionMat,2);
recall = @(confusionMat) diag(confusionMat)./sum(confusionMat,1)';
f1Scores = @(confusionMat) 2*(precision(confusionMat).*recall(confusionMat))./(precision(confusionMat)+recall(confusionMat));
meanPrecision = @(confusionMat) mean(precision(confusionMat));
meanRecall = @(confusionMat) mean(recall(confusionMat));
meanF1 = @(confusionMat) mean(f1Scores(confusionMat));
fprintf('Precision: %.4f\n',precision(confusionMat))
fprintf('Recall: %.4f\n',recall(confusionMat))
fprintf('F1 Scores: %.4f\n',f1Scores(confusionMat))
fprintf('Mean Precision: %.4f\n',meanPrecision(confusionMat))
fprintf('Mean Recall: %.4f\n',meanRecall(confusionMat))
fprintf('Mean F1 Score: %.4f\n',meanF1(confusionMat))
% === ROC and AUC
category = {'Control','Pneumonia'};
figure
for idx = 1:length(category)
    [X,Y,T,AUC] = perfcurve(cellstr(Label),PredProb(:,idx),category{idx});
    displayname = sprintf('%s - AUC: %.4f',category{idx},AUC);
    plot(X,Y,'DisplayName',displayname)
    hold on
end
legend('Location','SouthEast')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification (Ensemble Testing)')
% figureName = sprintf('Ensemble_Testing_ROC.png');
% saveas(gcf,fullfile(outputFolder,figureName))
