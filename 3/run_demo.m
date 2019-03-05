
function run_demo()

close all
clear all
clc

[H,r,c] = harris_corner_detector("./person_toy/00000001.jpg",1,3,3,0.0001);

%If want to plot with less cloudy circles, then only plot a subset of
%circles, change the plot from line 49 to line 52 in harris_corner_detector

end