
run_demo()

function [H,r,c] = harris(img,sigma,kernel_size,window_size,threshold)
    original = img;
    img = im2double(rgb2gray(img));
    sobel_x = double([1 0 -1; 2 0 -2; 1 0 -1]);
    sobel_y = double([1  2 1; 0 0 0; -1 -2 -1]);

    I_x = conv2(sobel_x,img);
    I_y = conv2(sobel_y,img);
    G = fspecial('gaussian',kernel_size,sigma);
    I_x_2 = I_x .* I_x;
    A= conv2(I_x_2,G);
    I_y_2 = I_y .* I_y;
    C = conv2(I_y_2,G);
    I_x_y = I_x .* I_y;
    B = conv2(I_x_y,G);
    H = (A .* C - B.^2 ) - 0.04 * (A + C).^2;

    r = [];
    c = [];
    y_max = size(H,1);
 
    x_max = size(H,2);

    if mod(window_size,2) == 1

        window_step = (window_size - 1) / 2;

        for y = 15:(y_max-15)

            for x = 15:(x_max-15)

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

                    
                    a = 10;
                    c = [c x];

                end

            end

        end

    end
end


function run_demo()

    close all
    clear all
    clc
    angle = 0;
    threshold = 0.1; 
    
    original = imread("./person_toy/00000001.jpg");
    next= imread("./person_toy/00000002.jpg");
    img = imrotate(original,angle);
   
    [~,r,c] = harris(img,1,3,3,threshold);

    if mod(angle,90) == 0
        marksize = 7;
    else
        marksize = 3;
    end

  
    %subplot(1,2,1)
    %imshow(img)
    %hold on 
    %plot(c,r, 'o', 'MarkerSize', marksize,'LineWidth', 0.000001);

    tracker_LK(original, next, c, r)

end

function [] = tracker_LK(image1, image2, c, r)
      f = figure()
    im1t = im2double(rgb2gray(image1));
    im1 = im1t; 
    im2 = im2double(rgb2gray(image2));
    w = 15;

    Ix_m = conv2(im1,[-1 1; -1 1], 'valid');
    Iy_m = conv2(im1, [-1 -1; 1 1], 'valid');
    It_m = conv2(im1, ones(2), 'valid') + conv2(im2, -ones(2), 'valid');

    u = zeros(length(c));
    v = zeros(length(c));
    
    imshow(image2)
    hold on
    for k = 1:length(c)
          i = c(k);
          j = r(k);
         
          plot(i,j, 'o');
          Ix = Ix_m(i-w:i+w, j-w:j+w);
          Iy = Iy_m(i-w:i+w, j-w:j+w);
          It = It_m(i-w:i+w, j-w:j+w);

          Ix = Ix(:);
          Iy = Iy(:);
               
          A = [Ix Iy]; 
          b = -It(:); 
             
          nu = pinv(A)*b; 
          
          u(k)=nu(1);
          v(k)=nu(2);
          
          x1 = [i i+u(k)];
          y1 = [j j+v(k)];
          q = quiver( x1(1),y1(1),x1(2)-x1(1),y1(2)-y1(1),200 )
          q.LineWidth = 1;
          q.MaxHeadSize = 1; 
          
          hold on
          
          
    end
    savefig("1.fig")
end