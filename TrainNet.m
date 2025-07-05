%% deepfish
%parpool
%clear
MainDir = fullfile('dataset/DeepFish/');
FishDir = fullfile(MainDir, 'Fish');

% フォルダ内のテキストファイルを取得
folderList = dir(fullfile(FishDir));

% テーブルを初期化
TrainData = table();
ValData = table();


% 各テキストファイルを処理
for i = 3:numel(folderList)
    TrainList = dir(fullfile(FishDir, folderList(i).name, 'train/', '*.txt'));
    ValList = dir(fullfile(FishDir, folderList(i).name, 'valid/', '*.txt'));
    TrainFolder(i, :) = fullfile(FishDir, folderList(i).name, 'train/');
    ValFolder(i, :) = fullfile(FishDir, folderList(i).name, 'valid/');
    for j = 1:numel(TrainList)
        Trainfile = fullfile(TrainList(j).folder, TrainList(j).name);
        Traindata = readmatrix(Trainfile);
        Traindata(:, [2 4]) = Traindata(:, [2 4])*1918;
        Traindata(:, [3 5]) = Traindata(:, [3 5])*1078;
        Traindata(:, 2) = Traindata(:, 2) - (Traindata(:,4))/2;
        Traindata(:, 3) = Traindata(:, 3) - (Traindata(:,5))/2;
        TraincellData = num2cell(cast(Traindata(:,2:end), "uint16")+1, [1 2]);
        Traintbl = table(TraincellData, 'VariableNames', {'fish'});
        TrainData = [TrainData; Traintbl];
    end
    for k = 1:numel(ValList)
        Valfile = fullfile(ValList(k).folder, ValList(k).name);
        Valdata = readmatrix(Valfile);
        Valdata(:, [2 4]) = Valdata(:, [2 4])*1918;
        Valdata(:, [3 5]) = Valdata(:, [3 5])*1078;
        Valdata(:, 2) = Valdata(:, 2) - (Valdata(:,4))/2;
        Valdata(:, 3) = Valdata(:, 3) - (Valdata(:,5))/2;
        ValcellData = num2cell(cast(Valdata(:,2:end), "uint16")+1, [1 2]);
        Valtbl = table(ValcellData, 'VariableNames', {'fish'});
        ValData = [ValData; Valtbl];
    end
end
imdsTrain = imageDatastore(string(TrainFolder(3:end, :)));
bldsTrain = boxLabelDatastore(TrainData);
imdsValidation = imageDatastore(string(ValFolder(3:end, :)));
bldsValidation = boxLabelDatastore(ValData);

trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
%%
validateInputData(trainingData);
validateInputData(validationData);


%% train
data = read(validationData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,"Rectangle",bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
%%
reset(trainingData);
inputSize = [416 416 3];
className = "fish";
rng("default")
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 9;
[anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);
area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");

anchors = anchors(idx,:);
% アンカーボックスの再設定
anchorBoxes = {anchors(1:4,:)
               anchors(5:9,:)};
detector = yolov4ObjectDetector("tiny-yolov4-coco",className,anchorBoxes,InputSize=inputSize);
%%
augmentedTrainingData = transform(trainingData,@augmentData);
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},"rectangle",data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,BorderSize=10)

%%
options = trainingOptions("adam", ...
    "GradientDecayFactor",0.9, ...
    "SquaredGradientDecayFactor",0.999, ...
    "InitialLearnRate",0.0001, ...       % 初期学習率の設定
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.1, ...
    "LearnRateDropPeriod",20, ...
    "MiniBatchSize",4, ...              % ミニバッチサイズの設定
    "L2Regularization",0.0001, ...       % L2 正則化係数
    "MaxEpochs",50, ...                 % 十分大きなエポック数に設定
    "BatchNormalizationStatistics","moving", ...
    "DispatchInBackground",true, ... 
    "ResetInputNormalization",false, ...
    "Shuffle","every-epoch", ...
    "Verbose", true, ...                % 詳細情報を表示
    "VerboseFrequency",50, ...
    "ValidationFrequency",50, ...       % 検証頻度
    "ValidationData",validationData, ...% 検証データの指定
    "ValidationPatience",10, ...        % 10回連続で改善がなければ早期終了
    "CheckpointPath",tempdir, ...       % チェックポイントの保存
    "Plots","training-progress", ...
    "ExecutionEnvironment","gpu");


% YOLO v4ディテクターのトレーニング
[detector,info] = trainYOLOv4ObjectDetector(augmentedTrainingData,detector,options);

%%
I = imread("7426_F1_f000003.jpg");
[bboxes,scores,labels] = detect(detector,I);
I = insertObjectAnnotation(I,"rectangle",bboxes,scores);
figure
imshow(I)

%%
function data = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.

data = cell(size(A));
for ii = 1:size(A,1)
    I = A{ii,1};
    bboxes = A{ii,2};
    labels = A{ii,3};
    sz = size(I);

    if numel(sz) == 3 && sz(3) == 3
        I = jitterColorHSV(I,...
            contrast=0.0,...
            Hue=0.1,...
            Saturation=0.2,...
            Brightness=0.2);
    end
    
    % Randomly flip image.
    tform = randomAffine2d(XReflection=true,Scale=[1 1.1]);
    rout = affineOutputView(sz,tform,BoundsStyle="centerOutput");
    I = imwarp(I,tform,OutputView=rout);
    
    % Apply same transform to boxes.
    [bboxes,indices] = bboxwarp(bboxes,tform,rout,OverlapThreshold=0.25);
    labels = labels(indices);
    
    % Return original data only when all boxes are removed by warping.
    if isempty(indices)
        data(ii,:) = A(ii,:);
    else
        data(ii,:) = {I,bboxes,labels};
    end
end
end

function data = preprocessData(data,targetSize)
% Resize the images and scale the pixels to between 0 and 1. Also scale the
% corresponding bounding boxes.

for ii = 1:size(data,1)
    I = data{ii,1};
    imgSize = size(I);
    
    bboxes = data{ii,2};

    I = im2single(imresize(I,targetSize(1:2)));
    scale = targetSize(1:2)./imgSize(1:2);
    bboxes = bboxresize(bboxes,scale);
    
    data(ii,1:2) = {I,bboxes};
end
end