
function run_demo()

close all
clear all
clc


%origineel
sigma = 1;
gauss_kernel_size = 3;
window_size = 3;
angle = 0;
threshold = 0.042; %0.042 %0.0002968 % 0.0001 
%Pong 0.192 and toy 0.042

original = imread("./person_toy/00000001.jpg");
%original = imread("./pingpong/0000.jpeg");
img = imrotate(original,angle);

[H,r,c] = harris_corner_detector(img,sigma,gauss_kernel_size,window_size,threshold);

if mod(angle,90) == 0
    marksize = 9;
else
    marksize = 3;
end


figure(1),imshow(img),title("Corner points on " + num2str(angle) + " degrees rotation of image","FontSize",13);
hold on 
plot(c,r, 'o', 'MarkerSize', marksize,'LineWidth', 0.000001);
hold off

%marksize = 3;
%figure(2),imshow(img),title("Corner points on " + num2str(angle) + " degrees rotation of image","FontSize",13);
%hold on 
%plot(c,r, 'o', 'MarkerSize', marksize,'LineWidth', 0.000001);
%hold off
%camroll(-angle)



end