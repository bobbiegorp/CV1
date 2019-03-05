

function [H,r,c] = harris_corner_detector(img,sigma,kernel_size,window_size,threshold)
    
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
    
    
    y_range = size(H,1);
    x_range = size(H,2);
    r = [];
    c = [];
    
    for y = y_range
        for x = x_range
            check_value = H(y,x);
            dominate_neighbours = true; 
            
            if check_value <= threshold
                continue
            end
            
            % Center and step size to left, right, top, bottom,
            % (window_size - 1) / 2
            %Assuming uneven number n to center around pixel, and ignore
            %outside images, zero padding effect
            for window_step = 1:((window_size-1)/2)
                x_plus_window = x + window_step;
                y_plus_window = y + window_step;
                x_min_window = x - window_step;
                y_min_window = y - window_step;
                top_left_diag = [y_min_window,x_min_window,];
                top_right_diag = [y_min_window,x_plus_window];
                bottom_right_diag = [y_plus_window,x_plus_window,];
                bottom_left_diag = [y_plus_window,x_min_window];
                
                l = [ [y x_plus_window] [y_plus_window x] [y x_min_window],[y_min_window x],top_left_diag,top_right_diag,bottom_right_diag,bottom_left_diag];
                
                for index = 1:2:length(l) 
                    window_y = l(index);
                    window_x = l(index + 1);
                    
                    dominate_neighbours = valid_range_or_dominate(window_y ,window_x,x_range,y_range,H,check_value);
                    if not(dominate_neighbours) 
                        break
                    end
                end 
                
                if not (dominate_neighbours)
                   break 
                end
               
            end
           
            if dominate_neighbours
                r = [r y];
                c = [c x];
            end
            
        end
    end
    
    
end


function dominant = valid_range_or_dominate(y_window,x_window,x_max,y_max,H,check_value)

    dominant = true;
    
    if x_window > 1 && x_window < x_max && y_window > 1 && y_window < y_max
        other_value = H(y_window,x_window);
        if check_value <= other_value
            dominant = false;
        end
        
    end

end

%[h,r,c] = harris_corner_detector("./person_toy/00000001.jpg",1,7,3,0.01)

