function G = gauss1D( sigma , kernel_size )
    G = zeros(1, kernel_size);
    if mod(kernel_size, 2) == 0
        error('kernel_size must be odd, otherwise the filter will not have a center to convolve on')
    end
    %% solution
    for i = 1:kernel_size
        x = -floor(kernel_size/2) + (i-1);
        G(i) = exp(-(x^2)/(2 * sigma^2))/(sigma*sqrt(2 * pi));
    end
    
    sum_g = sum(G);

    G = G ./ sum_g;

end
