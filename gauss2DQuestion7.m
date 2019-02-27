function [result] = gauss2D( sigma , kernel_size ) 
    %Q7.4
    % Generate horizontal and vertical coordinates, where the origin is in the middle
    ind = -floor(kernel_size/2) : floor(kernel_size/2);
    [X Y] = meshgrid(ind, ind);
    
    % Create Gaussian Mask
    h = exp(-(X.^2 + Y.^2) / (2*sigma*sigma));
    
    % Normalize so that total area (sum of all weights) is 1
    h = h / sum(h(:));
    result = h;
end


