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
end
