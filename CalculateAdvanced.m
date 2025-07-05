%% 物体検出モデルの評価（GPU対応・順次処理版）
try
    gpuDevice(1); % GPUデバイスの選択
    useGPU = true;
    fprintf('GPUを使用: %s\n', gpuDevice.Name);
catch
    useGPU = false;
    warning('GPUが利用できないか、設定に失敗しました。CPUを使用します。');
end

% タイマー開始
tic;

% 検出閾値の範囲設定
%thresholds = [0.01,0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7];
thresholds = [0.01,0.03,0.05,0.07, 0.09, 0.1, 0.12, 0.14,0.16,0.18];
numThresholds = length(thresholds);

% 結果を保存する配列を初期化
APValues = zeros(1, numThresholds);
AvgPrecisionValues = zeros(1, numThresholds);
AvgRecallValues = zeros(1, numThresholds);
F1ScoreValues = zeros(1, numThresholds);
PrecisionValsCell = cell(1, numThresholds);
RecallValsCell = cell(1, numThresholds);

% GPUを直接使用
if useGPU && isprop(detector, 'ExecutionEnvironment')
    detector.ExecutionEnvironment = 'gpu';
    fprintf('検出器をGPUモードに設定しました。\n');
end

for i = 1:numThresholds
    threshold = thresholds(i);
    fprintf('閾値 %.2f での評価を開始...\n', threshold);
    
    % 検出の実行（GPU利用可能ならGPUで実行）
    if useGPU
        try
            detectionResults = detect(detector, validationData, 'Threshold', threshold, 'ExecutionEnvironment', 'gpu');
        catch
            detectionResults = detect(detector, validationData, 'Threshold', threshold);
        end
    else
        detectionResults = detect(detector, validationData, 'Threshold', threshold);
    end
    
    % 評価指標の計算
    metrics = evaluateObjectDetection(detectionResults, validationData);
    
    % 評価するクラスの選択
    classID = 1;
    
    APValues(i) = metrics.ClassMetrics.mAP(classID);
    RecallValsCell{i} = metrics.ClassMetrics.Recall{classID};
    PrecisionValsCell{i} = metrics.ClassMetrics.Precision{classID};
    AvgPrecisionValues(i) = mean(PrecisionValsCell{i}, 'omitnan');
    AvgRecallValues(i) = mean(RecallValsCell{i}, 'omitnan');
    
    % F1スコアの計算
    F1ScoreValues(i) = 2 * (AvgPrecisionValues(i) * AvgRecallValues(i)) / ...
                       (AvgPrecisionValues(i) + AvgRecallValues(i));
    
    fprintf('閾値 %.2f での評価が完了しました。AP: %.5f, F1: %.5f\n', ...
        threshold, APValues(i), F1ScoreValues(i));
end

% 最良のF1スコアを持つ結果のインデックスを特定
[bestF1, bestIdx] = max(F1ScoreValues);
bestThreshold = thresholds(bestIdx);

processingTime = toc;

%% 結果の表示
fprintf('\n==== 評価結果のサマリー（処理時間: %.2f秒） ====\n', processingTime);
fprintf('最適な閾値: %.2f (F1スコア: %.5f)\n\n', bestThreshold, bestF1);

% 全ての閾値の結果をテーブル形式で表示
resultTable = table(thresholds', APValues', AvgPrecisionValues', ...
                    AvgRecallValues', F1ScoreValues', ...
                    'VariableNames', {'閾値', 'AP', '平均適合率', '平均再現率', 'F1スコア'});
disp(resultTable);

% 最良の結果の詳細表示
fprintf('\n--- 最良閾値 (%.2f) での詳細評価指標 ---\n', bestThreshold);
fprintf('平均適合率 (AP): %.5f\n', APValues(bestIdx));
fprintf('平均適合率 (Mean Precision): %.5f\n', AvgPrecisionValues(bestIdx));
fprintf('平均再現率 (Mean Recall): %.5f\n', AvgRecallValues(bestIdx));
fprintf('F1スコア: %.5f\n', F1ScoreValues(bestIdx));

% 最良の結果の適合率-再現率曲線をプロット
figure;
plot(RecallValsCell{bestIdx}, PrecisionValsCell{bestIdx}, 'LineWidth', 2);
grid on;
title(sprintf('閾値 %.2f での適合率-再現率曲線 (AP = %.5f)', bestThreshold, APValues(bestIdx)));
xlabel('再現率 (Recall)');
ylabel('適合率 (Precision)');
axis([0 1 0 1]);

% 全ての閾値の適合率-再現率曲線を1つのグラフにプロット
figure;
hold on;
colormap = parula(numThresholds);
for i = 1:numThresholds
    plot(RecallValsCell{i}, PrecisionValsCell{i}, 'LineWidth', 1.5, 'Color', colormap(i,:));
end
grid on;
legend(cellstr(num2str(thresholds', '閾値 = %.2f')), 'Location', 'southwest');
title('異なる閾値での適合率-再現率曲線の比較');
xlabel('再現率 (Recall)');
ylabel('適合率 (Precision)');
axis([0 1 0 1]);
hold off;

% 元の閾値での結果も出力
fprintf('\n--- 元の閾値 (0.5) での評価指標 ---\n');
origThresholdIdx = find(thresholds == 0.5);
if ~isempty(origThresholdIdx)
    fprintf('クラス %d の平均適合率(AP): %.5f\n', 1, APValues(origThresholdIdx));
    fprintf('平均適合率 (Mean Precision): %.5f\n', AvgPrecisionValues(origThresholdIdx));
    fprintf('平均再現率 (Mean Recall): %.5f\n', AvgRecallValues(origThresholdIdx));
    fprintf('F1スコア: %.5f\n', F1ScoreValues(origThresholdIdx));
else
    fprintf('閾値 0.5 での結果は計算されていません。\n');
end

% 元のように閾値0.5での適合率-再現率曲線もプロット
if ~isempty(origThresholdIdx)
    figure;
    plot(RecallValsCell{origThresholdIdx}, PrecisionValsCell{origThresholdIdx}, 'LineWidth', 1.5);
    grid on;
    title(['Average Precision = ', num2str(APValues(origThresholdIdx), '%.5f')]);
    xlabel('Recall');
    ylabel('Precision');
    axis([0 1 0 1]);
end