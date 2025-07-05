%% 検証データの読み込みとdatastoreの作成
MainDir = fullfile('dataset/DeepFish/');
FishDir = fullfile(MainDir, 'Fish');

% フォルダ内の魚種フォルダリストを取得（'.'と'..'を除外）
folderList = dir(FishDir);

% 検証用バウンディングボックス情報を格納するテーブルとフォルダパスの初期化
ValData = table();
ValFolder = strings(numel(folderList),1);

% 各魚種フォルダ内の検証データ（valid）を処理
for i = 3:numel(folderList)
    % 検証用のテキストファイルリストを取得
    ValList = dir(fullfile(FishDir, folderList(i).name, 'valid', '*.txt'));
    % 対応する検証画像フォルダのパスを保存
    ValFolder(i) = fullfile(FishDir, folderList(i).name, 'valid');
    
    % 各テキストファイルを読み込み、座標変換を実施
    for k = 1:numel(ValList)
        Valfile = fullfile(ValList(k).folder, ValList(k).name);
        Valdata = readmatrix(Valfile);
        
        % ※ ここでは例として画像サイズ1918×1078でスケーリング後，
        %    バウンディングボックスの中心座標→左上座標への変換を実施
        Valdata(:, [2 4]) = Valdata(:, [2 4]) * 1918;
        Valdata(:, [3 5]) = Valdata(:, [3 5]) * 1078;
        Valdata(:, 2) = Valdata(:, 2) - (Valdata(:,4))/2;
        Valdata(:, 3) = Valdata(:, 3) - (Valdata(:,5))/2;
        
        % バウンディングボックス情報（2列目以降）をセル配列へ変換（+1で補正）
        ValcellData = num2cell(cast(Valdata(:,2:end), "uint16") + 1, [1 2]);
        % "fish"クラスとしてテーブルに格納
        Valtbl = table(ValcellData, 'VariableNames', {'fish'});
        ValData = [ValData; Valtbl];
    end
end

% 検証画像フォルダのうち、最初の2要素は'.','..'となるため除外
imdsValidation = imageDatastore(string(ValFolder(3:end)));

% テーブル形式のラベル情報からboxLabelDatastoreを作成
bldsValidation = boxLabelDatastore(ValData);

% 画像とラベル情報を結合して検証用データセットを作成
validationData = combine(imdsValidation, bldsValidation);
