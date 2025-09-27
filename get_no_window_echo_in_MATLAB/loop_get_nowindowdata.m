% 循环计算得到所有的去窗结果
clc;close all;clear all;

% 定义输入文件夹路径
input_folder = 'sample'; % mat文件路径



files = dir(fullfile(input_folder, '*.mat'));

hWaitBar = waitbar(0, 'Initializing...');
total_files = length(files);

for i = 1:length(files)
    if ~isdir(fullfile(input_folder, files(i).name)) 
      
        current_path = fullfile(input_folder, files(i).name);
        load(current_path);

        img2Spec = @(img) fty(ftx(img));
        spec2Img = @(spec) iftx(ifty(spec)); 
        
        numValid = 102;
        tmp = img2Spec(complex_img);
        specRmZero = tmp(end/2-numValid/2+1:end/2+numValid/2, end/2-numValid/2+1:end/2+numValid/2); % 去除多余零点
        win2d = taylorwin(size(specRmZero, 1), 4, -35) * taylorwin(size(specRmZero, 2), 4, -35).';  % 二维的窗函数
        specRmZeroAndWin = specRmZero ./ win2d;     % 去窗
        preImg = spec2Img(specRmZeroAndWin);
        
        % 将新变量保存回原.mat文件（使用原始文件名并追加保存）
        save(fullfile(input_folder, files(i).name), 'specRmZeroAndWin', '-append');
        
      

        progress_fraction = num2str(i/total_files);
        waitbar(i/total_files, hWaitBar, ['Processing file: ', progress_fraction, ' completed']);
        

        
    end
end
close(hWaitBar); 