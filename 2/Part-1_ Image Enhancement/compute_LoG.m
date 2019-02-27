function imOut = compute_LoG(image, LOG_type)
% Get image
img = im2double(imread(image));

switch LOG_type
    case 1
        %method 1
        method = "Smoothing with Gaussian, then take Laplacian";
        h = fspecial('gaussian',5,0.5);
        img_gaus = conv2(h,img);
        k = fspecial('laplacian');
        imOut = conv2(k,img_gaus);
    case 2
        %method 2
        method = "Laplacian of Gaussian kernel";
        k = fspecial('log',5,0.5);
        imOut = conv2(k,img);
    case 3
        %method 3
        method = "Difference of two Gaussians";
        h = fspecial('gaussian',5,1);
        img_intermediate1 = conv2(h,img);
        k = fspecial('gaussian',5,1.6);
        img_intermediate2 = conv2(k,img);
        
        
        imOut = img_intermediate1 - img_intermediate2;
end

figure, imshow(imOut, []), title(method);
end

