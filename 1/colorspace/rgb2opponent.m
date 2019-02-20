function [output_image] = rgb2opponent(input_image)
% converts an RGB image into opponent color space
[R, G, B] = getColorChannels(input_image);

O1 = 1/sqrt(2) * (R - G);
O2 = 1/sqrt(6) * (R + G - 2*B);
O3 = 1/sqrt(3) * (R + G + B);

output_image = cat(3, O1, O2, O3);
end

