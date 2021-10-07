close all;clear;clc;
finf = dir('/home/john/Desktop/REFUGE/test/original/*.jpg');
long = length(finf);
for k = 1:long
    pic_name = finf(k).name;
    name = split(pic_name, ".");
    ori_name = name{1};
    ori = imread(['/home/john/Desktop/REFUGE/test/original/' ori_name '.jpg']);
    LDF = imread(['/home/john/Desktop/REFUGE/test/DOG/' ori_name '.jpg']);
    [m,n,z] = size(ori);
    I = double(ori(:,:,2));
    Ivessel=FrangiFilter2D(I)*255;
    [x,y]=find(Ivessel>4);
    number = length(x);
    for i=1:number
        LDF(x(i),y(i),:)=0;
    end
%     figure,imshow(LDF)
    imwrite(LDF,['/home/john/Desktop/REFUGE/test/LDF Blood/' ori_name '.jpg']);
end

% I = imread('/home/john/Desktop/REFUGE/training/Blood/V0002.bmp');
% Y3= medfilt2(I,[3 3]); 
% figure,imshow(Y3);