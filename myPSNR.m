function [ PSNR ] = myPSNR( orig_image, approx_image )
%This code calculates and prints the PSNR of two images (an original image and the
%approximation image. 

%%%% Comment the following line from approx_image (as shown in this example), when using denoise.m , because the images are
%%%% already read. 
orig_image = imread(orig_image);% approx_image = imread(approx_image);
orig_image = double(orig_image);
approx_image = double(approx_image);
ImaxRoot = max(max(orig_image)).^2;
d = sum((orig_image(:)-approx_image(:)).^2) / prod(size(orig_image));
PSNR = 10*log10(ImaxRoot/d)
end

