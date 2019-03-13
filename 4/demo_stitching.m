function demo_stitching()

run("./vlfeat-0.9.21/toolbox/vl_setup")

close all
clear all
clc

left = imread("./left.jpg");
right = imread("./right.jpg");

%%%% temp transform to gray, later color should be possible as well
left = single(rgb2gray(left));
right = single(rgb2gray(right));
%%%%

stitched_image = stitch(left, right);
figure; imshow(stitched_image,[]);

end