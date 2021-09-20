function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14'});
predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_13;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false];
template = templateTree(...
    'MinLeafSize', 3, ...
    'NumVariablesToSample', 12);
regressionEnsemble = fitrensemble(...
    predictors, ...
    response, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', 472, ...
    'Learners', template, ...
    'LearnRate', 0.7820958418923589);
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));
trainedModel.RegressionEnsemble = regressionEnsemble;
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14'});
predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_13;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false];
validationPredictions = predict(trainedModel.RegressionEnsemble, predictors);
validationRMSE = sqrt(resubLoss(trainedModel.RegressionEnsemble, 'LossFun', 'mse'));
