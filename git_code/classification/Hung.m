close all;clear;clc;
path='/home/john/Desktop/REFUGE/test/original/';
mask_path='/home/john/Desktop/REFUGE/test/OD/';
finf = dir([path '*.jpg']);
long = length(finf);
for k=1:long
    pic_name = finf(k).name;
    ori_name = pic_name(1:end-4);
    I = imread([path ori_name '.jpg']);
    mask = 255-imread([mask_path ori_name '.bmp']);
    [x,y] = find(mask==255);
    x = round(sum(x(:))/length(x));
    y = round(sum(y(:))/length(y));
    
    if (y-200)<=0
        final = I(x-300:x+299,1:y+499,:);
    else
        final = I(x-300:x+299,y-200:y+499,:);
    end
    imwrite(final,['/media/john/SP PHD U3/Glau/1200/' ori_name '.jpg']);
%     figure,imshow(final)
%     figure,imshow(I)
%     hold on
%     plot(y,x,'r*')
%     figure,imshow(mask)
end