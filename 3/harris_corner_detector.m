

function [H,r,c] = harris_corner_detector(img,sigma,kernel_size,window_size,threshold)

    original = imread(img);
    img = im2double(rgb2gray(imread(img)));
   
    %sigma = 1;
    %kernel_size = 7;
    
    %First order Gaussian filter approximation
    sobel_x = double([1 0 -1; 2 0 -2; 1 0 -1]);
    sobel_y = double([1  2 1; 0 0 0; -1 -2 -1]);

    %Calculate convolution with First order Gaussian 
    I_x = conv2(sobel_x,img);
    I_y = conv2(sobel_y,img);
    
    G = fspecial('gaussian',kernel_size,sigma);
    
    I_x_2 = I_x .* I_x;
    A= conv2(I_x_2,G);
    
    I_y_2 = I_y .* I_y;
    C = conv2(I_y_2,G);
    
    I_x_y = I_x .* I_y;
    B = conv2(I_x_y,G);
    
    %Cornerness H, or response of each pixel of the detector R in slides
    H = (A .* C - B.^2 ) - 0.04 * (A + C).^2;
    
    figure(1), imshow(A,[]), title("A");
    figure(2), imshow(B, []), title("B");
    figure(3), imshow(C, []), title("C");
    figure(4), imshow(H, []), title("H");
    figure(5), imshow(I_x, []), title("I_x");
    figure(6), imshow(I_y, []), title("I_y");
     
    [r1,c1] = find(H > threshold);
    lst = [];
    range = floor(window_size/2);
    
    for n = 1 : length(r1)
        i = r1(n);
        j = c1(n);
        
        window = H( (max(1, i - range) : min(i + range, length(H(:,1)))) , (max(1, j - range) : min(j + range, length(H(1,:)))) );
        [r2,~] = find(window > H(i,j));
        
        if not(isempty(r2))
            lst = [lst n];
        end    
    end
    
    r = zeros(length(lst),1);
    c = zeros(length(lst),1);
    
    for i = 1: length(lst)
        r(i) = r1(lst(i));
        c(i) = c1(lst(i));
    end
    
    
    figure(7),imshow(original);
    
    %axis on
    hold on
   
    %Plot all
    figure(8), plot(c,r, 'o', 'MarkerSize', 7,'LineWidth', 0.000001);
    
    %Plot intervals to make it less cloudy for visualization
    %plot(c,r,'o','MarkerIndices',1:7:length(r))

    hold off
   
    
end
%harris_corner_detector("./person_toy/00000001.jpg",1,9,3,0.00013);
%[h,r,c] = harris_corner_detector("./person_toy/00000001.jpg",1,9,3,0.00001);
%harris_corner_detector("./person_toy/00000001.jpg",1,3,3,0.00013);
