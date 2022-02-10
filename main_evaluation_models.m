close all; clear; clc;
set(0,'defaultAxesFontSize',13)
EvaluationOutputFolder = 'Result/Evaluation/';
mkdir(EvaluationOutputFolder)
EnsembleOutputFolder = 'Result/Ensemble/';
mkdir(EnsembleOutputFolder)

%% === < model checking > ===
mat_dir = dir('Result/*.mat');

%% === < evaluations of models > ===
load(fullfile(mat_dir(1).folder,mat_dir(1).name))
evaluation_tbl = zeros(length(mat_dir),7);
YPredProbAll = zeros(size(YPredProb));
fig = figure();
fig.Position(3) = 1.45*fig.Position(3);
for idx_model = 1:length(mat_dir)
    load(fullfile(mat_dir(idx_model).folder,mat_dir(idx_model).name))
    
    confusionMat = confusionmat(YValidation,YPred);
    
    tp = confusionMat(4);
    tn = confusionMat(1);
    fp = confusionMat(2);
    fn = confusionMat(3);
    
    accuracy = sum(YPred == YValidation)/numel(YValidation);
    precision = tp / ( tp + fp );
    recall = tp / ( tp + fn );
    specificity = tp / ( tn + fp );
    f1Score = 2*precision*recall ./ (precision+recall);
    [X,Y,T,auc] = perfcurve(cellstr(YValidation),YPredProb(:,2),'Pneumonia');
    displayname = sprintf('%s (AUC: %.4f)',modelName,auc);
    plot(X,Y,'DisplayName',displayname)
    hold on
    
    modelNameList{idx_model,1} = modelName;
    evaluation_tbl(idx_model,2) = accuracy;
    evaluation_tbl(idx_model,3) = precision;
    evaluation_tbl(idx_model,4) = recall;
    evaluation_tbl(idx_model,5) = specificity;
    evaluation_tbl(idx_model,6) = f1Score;
    evaluation_tbl(idx_model,7) = auc;
    
    fprintf('Model: %s\n',modelName)
    fprintf('Accuracy: %.4f\n',accuracy)
    fprintf('Precision: %.4f\n',precision)
    fprintf('Recall: %.4f\n',recall)
    fprintf('Specificity: %.4f\n',specificity)
    fprintf('F1 Scores: %.4f\n',f1Score)
    fprintf('AUC: %.4f\n',auc)
    fprintf('\n')
    
    YPredProbAll = YPredProbAll + (1/length(mat_dir))*YPredProb;
end
legend('Location','NorthEastoutside')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification')
grid on
figureName = sprintf('All Models ROC.png');
saveas(gcf,fullfile(EvaluationOutputFolder,figureName))

tbl = array2table(evaluation_tbl, ...
    'VariableNames',{'Model','Accuracy','Precision','Recall','Specificity','F1 Score','AUC'});
tbl.Model = modelNameList;
fileName = sprintf('All Models Evaluation.xlsx');
writetable(tbl,fullfile(EvaluationOutputFolder,fileName))

%% === < evaluations of ensemble model > ===
[YPredProbAll_value,YPredProbAll_loc] = max(YPredProbAll,[],2);
YPredProbAll_type_list = {};
% Control / Pneumonia
for idx = 1:length(YPredProbAll_loc)
    YPredProbAll_loc_idx = YPredProbAll_loc(idx);
    if YPredProbAll_loc_idx == 1
        YPredProbAll_type = 'Control';
    elseif YPredProbAll_loc_idx == 2
        YPredProbAll_type = 'Pneumonia';
    else
        error('Wrong!!!');
    end
    YPredProbAll_type_list{idx,1} = YPredProbAll_type;
end

Pred = categorical(YPredProbAll_type_list);

% === ground truth
Label = YValidation;
% === confusion matrix
confusionMat = confusionmat(Label,Pred);
figure
plotconfusion(Label,Pred)
title('Validation Confusion Matrix (Ensemble)')
figureName = sprintf('Ensemble_Validation_ConfusionMatrix.png');
saveas(gcf,fullfile(EnsembleOutputFolder,figureName))
% === ROC and AUC
figure
[X,Y,T,AUC] = perfcurve(cellstr(Label),YPredProbAll(:,2),'Pneumonia');
plot(X,Y)
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification (Ensemble Validation)')
grid on
figureName = sprintf('Ensemble_Validation_ROC.png');
saveas(gcf,fullfile(EnsembleOutputFolder,figureName))
% === ROC and AUC
tp = confusionMat(4);
tn = confusionMat(1);
fp = confusionMat(2);
fn = confusionMat(3);

accuracy = sum(Pred == Label)/numel(Label);
precision = tp / ( tp + fp );
recall = tp / ( tp + fn );
specificity = tp / ( tn + fp );
f1Score = 2*precision*recall ./ (precision+recall);

fprintf('Ensemble\n')
fprintf('Accuracy: %.4f\n',accuracy)
fprintf('Precision: %.4f\n',precision)
fprintf('Recall: %.4f\n',recall)
fprintf('Specificity: %.4f\n',specificity)
fprintf('F1 Scores: %.4f\n',f1Score)
fprintf('AUC: %.4f\n',auc)
fprintf('\n')

fprintf('Finish!!!\n')
