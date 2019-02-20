function [output_image] = rgb2grays(input_image)
% converts an RGB into grayscale by using 4 different methods
[R,G,B] = getColorChannels(input_image);

% lightness method
maximum = max(input_image, [], 3);
minimum = min(input_image, [],3);

output_lightness = 0.5 * (maximum + minimum);

% average method
output_avg = mean(input_image, 3);

% luminosity method
output_luminosity = 0.21 * R + 0.72 * G + 0.07 * B;

% built-in MATLAB function 
output_matlab = rgb2gray(input_image);

% combine results from the different method to one matrix
output_image = cat(3, output_lightness, output_avg, output_luminosity, output_matlab);
end