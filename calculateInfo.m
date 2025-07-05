%% 物体検出モデルの評価(AIで生成したので，ちゃんと動くかわからない)
% 学習済みモデルを使用して検証データセット上で物体検出を実行し、
% 性能指標を算出するプログラム

% モデルによる物体検出の実行（検出閾値 0.5）
detectionResults = detect(net, validationData, 'Threshold', 0.5);

% 検出結果の評価指標を計算
metrics = evaluateObjectDetection(detectionResults, validationData);

% 評価するクラスの選択（ここではクラス1）
classID = 1;

% 平均適合率（Average Precision）の取得
ap = metrics.ClassMetrics.mAP(classID);

% 再現率と適合率の値を取得
recallVals    = metrics.ClassMetrics.Recall{classID};
precisionVals = metrics.ClassMetrics.Precision{classID};

%% 追加性能指標の計算
% 平均適合率（NaN値を除外）
avgPrecision = mean(precisionVals, 'omitnan');

% 平均再現率（NaN値を除外）
avgRecall = mean(recallVals, 'omitnan');

% F1スコア（適合率と再現率の調和平均）
f1Score = 2 * (avgPrecision * avgRecall) / (avgPrecision + avgRecall);

%% 結果の表示
% 基本指標の表示
fprintf('クラス %d の平均適合率(AP): %.5f\n', classID, ap);

% 適合率-再現率曲線のプロット
figure
plot(recallVals, precisionVals, 'LineWidth', 1.5);
grid on
title(['Average Precision = ', num2str(ap, '%.5f')]);
xlabel('Recall');
ylabel('Precision');
axis([0 1 0 1]);  % 軸の範囲を0〜1に設定

% 追加指標の表示
fprintf('\n--- 詳細な評価指標 ---\n');
fprintf('平均適合率 (AP): %.5f\n', ap);
fprintf('平均適合率 (Mean Precision): %.5f\n', avgPrecision);
fprintf('平均再現率 (Mean Recall): %.5f\n', avgRecall);
fprintf('F1スコア: %.5f\n', f1Score);