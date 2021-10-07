close all;clear;clc;
finf_ori = dir('D:\od oc\train\ROI data\*.jpg');
long = length(finf_ori);
for k=60
    %:long
    pic_name = finf_ori(k).name;
    name = split(pic_name, ".");
    ori_name = name{1};
    I = imread(['D:\od oc\train\ROI data\' ori_name '.jpg']);
    figure,imshow(I)
%     h = imrect;
%     pos = getPosition(h);
%     pos = round(pos);
%     close(gcf);
%     I_box = I(pos(2):pos(2)+pos(4),pos(1):pos(1)+pos(3),:);
%     figure,imshow(I_box)
    sigma = 0.2;
    alpha = 0.3;
    B_speed = locallapfilt(I, sigma, alpha, 'NumIntensityLevels', 20);
    figure,imshow(B_speed)
%     I_box = double(I_box);
%     [r, c, ~] = size(I_box);
%     [idx, mp] = rgb2ind(I_box,32);
%     x = (1:c)-(c/2); 
%     y = (1:r)-(r/2); 
%     [xp, yp] = meshgrid(linspace(0,2*pi), linspace(0,400));
%     [xx, yy] = pol2cart(xp,yp); 
%     out = interp2(x, y, double(idx), xx, yy);
% %     imwrite(out, mp, 'result.png') 
%     figure,imshow(out)
end