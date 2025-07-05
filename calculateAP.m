detectionResults = detect(net, validationData, 'Threshold', 0.5);
metrics = evaluateObjectDetection(detectionResults, validationData);

classID = 1;
ap = metrics.ClassMetrics.mAP(classID);

recallVals    = metrics.ClassMetrics.Recall{classID};
precisionVals = metrics.ClassMetrics.Precision{classID};

% 結果の表示
fprintf("AP (class %d): %.5f\n", classID, ap);

figure
plot(recallVals, precisionVals);
grid on
title("Average Precision = " + ap);
xlabel("Recall");
ylabel("Precision");