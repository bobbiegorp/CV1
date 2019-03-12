function demo_stitching()

run("./vlfeat-0.9.21/toolbox/vl_setup")

close all
clear all
clc

left = imread("./left.jpg");
right = imread("./right.jpg");

stitched_image = stitch(left, right);
figure; imshow(stitched_image,[]);

end