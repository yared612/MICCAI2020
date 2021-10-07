close all;clear;clc;
path = 'I:\Glau\img\crop\train\glaucoma\';
save_path = 'I:\Glau\matching\train\glaucoma\';

firf = dir([path '*.jpg']);
long = length(firf);