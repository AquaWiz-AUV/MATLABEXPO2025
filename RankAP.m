modelsFolder = 'C:\Users\s-mat\AppData\Local\Temp\mat_models';
modelFiles = dir(fullfile(modelsFolder, '*.mat'));
results = struct();
k = 0; % 有効なモデル用のカウンタ

for i = 1:length(modelFiles)
    modelPath = fullfile(modelsFolder, modelFiles(i).name);
    loadedData = load(modelPath);
    
    if isfield(loadedData, 'net')
        net = loadedData.net;
        % 評価データとして validationData を使用
        detectionResults = detect(net, validationData);
        metrics = evaluateObjectDetection(detectionResults, validationData);
        classID = 1;
        ap = metrics.ClassMetrics.mAP(classID);
        
        k = k + 1;
        results(k).ModelName = modelFiles(i).name;
        results(k).AveragePrecision = ap;
        
        % Use modelFiles(i).name instead of loadedData
        fprintf('%s -> AP: %.5f\n', modelFiles(i).name, ap);
    else
        fprintf('Skipping %s (no "net" found)\n', modelFiles(i).name);
    end
end

% 有効な結果のみ、平均適合率 (AP) の降順でソートしてランキングを出力
[~, sortIdx] = sort([results.AveragePrecision], 'descend');
results = results(sortIdx);

disp('--- Average Precision Ranking ---');
for i = 1:length(results)
    fprintf('%d: %s -> AP: %.5f\n', i, results(i).ModelName, results(i).AveragePrecision);
end
