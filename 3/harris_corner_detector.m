

function [H,r,c] = harris_corner_detector(img,sigma,kernel_size,window_size,threshold)

    original = img;
    img = im2double(rgb2gray(img));
    
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
    
    %figure(1), imshow(A,[]), title("A");
    %figure(2), imshow(B, []), title("B");
    %figure(3), imshow(C, []), title("C");
    %figure(4), imshow(H, []), title("H");
    %figure(5), imshow(I_x, []), title("Image derivative: Ix");
    %figure(6), imshow(I_y, []), title("Image derivative: Iy");
    %imwrite (mat2gray(I_x), "toy_042_I_x.png");
    %imwrite (mat2gray(I_y), "toy_042_I_y.png");
     
    %[r,c] = find(H > threshold);
    r = [];
    c = [];
    y_max = size(H,1);
    x_max = size(H,2);

     % Has a center pixel
    if mod(window_size,2) == 1
        window_step = (window_size - 1) / 2;
        for y = 1:y_max
            for x = 1:x_max
                check_value = H(y,x);
                top_left_y = y - window_step;
                top_left_x = x - window_step;
                y_window_range = top_left_y + window_size - 1;
                x_window_range = top_left_x + window_size - 1;
                
                top_left_y = max(top_left_y,1);
                top_left_x = max(top_left_x,1);
                y_window_range = min(y_window_range,y_max); 
                x_window_range = min(x_window_range,x_max);
                
                neighbourhood = H(top_left_y:y_window_range, top_left_x:x_window_range);
                amount = sum(sum(neighbourhood >= check_value));
                if amount == 1 && check_value > threshold
                    r = [r y];
                    c = [c x];
                end
            end
        end
    end
    
    %figure(7),imshow(original),title("Corner points on original image","FontSize",17);
    
    %axis on
    %hold on;
    %Plot all
    %plot(c,r, 'o', 'MarkerSize', 7,'LineWidth', 0.000001);
    %hold off;
   
    
end


