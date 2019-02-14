function [output_image] = rgb2normedrgb(input_image)
% converts an RGB image into normalized rgb
[R, G, B] = getColorChannels(input_image);
total = double(sum(input_image, 3));

R = double(R) ./ total;
G = double(G) ./ total;
B = double(B) ./ total;

output_image = cat(3, R, G, B);
end

