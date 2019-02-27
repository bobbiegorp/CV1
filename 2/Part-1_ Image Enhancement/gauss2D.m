function G = gauss2D( sigma , kernel_size )
    %% solution
    G = zeros(kernel_size, kernel_size);
    x = gauss1D(sigma, kernel_size);
    y = gauss1D(sigma, kernel_size);
    for i = 1:kernel_size
        for j = 1:kernel_size
            G(i,j) = x(i) * y(j);
        end
    end
    
%Used this Gauss2D for question 7.4
%function [result] = gauss2D( sigma , kernel_size ) 
%    %Q7.4
%    % Generate horizontal and vertical coordinates, where the origin is in the middle
%    ind = -floor(kernel_size/2) : floor(kernel_size/2);
%    [X Y] = meshgrid(ind, ind);
    
    % Create Gaussian Mask
%    h = exp(-(X.^2 + Y.^2) / (2*sigma*sigma));
    
    % Normalize so that total area (sum of all weights) is 1
%    h = h / sum(h(:));
%    result = h;
%end
end
