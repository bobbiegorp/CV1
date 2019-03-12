

function demo_alignment()

% To use vl_sift one time, or permenantly add to startup.m file
%path =  <folder where vlfeat>
%run(path + "./vlfeat-0.9.21/toolbox/vl_setup")
run("/home/marvin/Documenten/Master_AI_leerjaar_1/Computer vision 1/vlfeat-0.9.21/toolbox/vl_setup")

close all
clear all
clc

image1 = imread("./boat1.pgm");
image2 = imread("./boat2.pgm");

matches = keypoint_matching(image1, image2);

amount_matches = 20;
n_repeat = 10;
RANSAC(image1,image2,matches,amount_matches,n_repeat);


end