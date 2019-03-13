
function [m1, m2, m3, m4, t1, t2] = RANSAC(original_image1,original_image2,matches,p_amount,n_repeat)

%Some preprocessing again to get the frames information
if size(original_image1,3) == 3
    image1 = single(rgb2gray(original_image1));
    image2 = single(rgb2gray(original_image2));
else
    image1 = single(original_image1);
    image2 = single(original_image2);
end 

[f,d] = vl_sift(image1);
[f2,d2] = vl_sift(image2);

best_inliers = -1;
for repeat = 1:n_repeat
    %Get p_amount of pairs
    perm_matches = randperm(size(matches,2));
    pair_index = perm_matches(1:p_amount);
    pair_indices = matches(:,pair_index);
    index_points_im1 = pair_indices(1,:);
    index_points_im2 = pair_indices(2,:);

    %Get the correspondin x and y for each image and put in vector
    in_x = f(1,index_points_im1);
    in_y = f(2,index_points_im1);
    out_x = f2(1,index_points_im2);
    out_y = f2(2,index_points_im2);

    %For each value in vector, stack in the structure of Ax = b
    b = [];
    A = [];
    for i = 1:length(out_x)
        b = [b; out_x(i);out_y(i)];
        A = [A; in_x(i), in_y(i), 0,0,1,0;0,0,in_x(i), in_y(i),0,1];
    end

    %Compute and assign parameters variables
    parameters_x = pinv(A) * b;

    %Compute amount of inliers
    size_x = size(original_image2,2);
    size_y = size(original_image2,1);
    [col, row] = meshgrid(1:size_x, 1:size_y);
    inliers = 0;
    inliers_set = [];
    for i = 1:size(matches,2)
        pair = matches(:,i);
        index_image1 = pair(1);
        index_image2 = pair(2);

        f1_info = f(:,index_image1);
        x = f1_info(1);
        y = f1_info(2);

        f2_info =  f2(:,index_image2);
        x2 = f2_info(1);
        y2 = f2_info(2); 

        %Transform keypoints of image 1 
        A_pixel = [x,y, 0,0,1,0;0,0,x,y,0,1];
        b_pixel = A_pixel * parameters_x;
        new_x = round(b_pixel(1));
        new_y = round(b_pixel(2));
        if new_x < 1
           new_x = 1;
        end
        if new_y < 1
           new_y = 1;
        end

        %Check if transformed coordinates are within radius of coordinates of
        %pair in image 2
        y_r = (row - y2).^2;
        x_r = (col - x2).^2;
        combined = x_r + y_r;
        
        matrix = combined <= 10.^2;
        if new_y < size(matrix,1) && new_y > 0 && new_x < size(matrix,2) && new_x > 0 
            if matrix(new_y,new_x) == 1 
                inliers = inliers + 1 ;
                inliers_set = [inliers_set; new_y new_x y2 x2];
            end 
        end
    end
    
    if inliers > best_inliers
       best_parameters_x = parameters_x;
       best_inliers = inliers;
       best_inliers_set = inliers_set;
    end
    
    if inliers > 0
       disp("Inliers: " + inliers)
    end 
    
end

m1 = best_parameters_x(1);
m2 = best_parameters_x(2);
m3 = best_parameters_x(3);
m4 = best_parameters_x(4);
t1 = best_parameters_x(5);
t2 = best_parameters_x(6);
disp(best_parameters_x);

%To create transformed image of image 1 using best parameters found, every pixel to transform
transformed_image = zeros(size(image1));
for y = 1:size(image1,1)
    for x = 1: size(image1,2)
        value = image1(y,x);
        A_pixel = [x,y, 0,0,1,0;0,0,x,y,0,1];
        b_pixel = A_pixel * best_parameters_x;
        new_x = round(b_pixel(1));
        new_y = round(b_pixel(2));
        if new_x < 1
           new_x = 1;
        end
        if new_y < 1
           new_y = 1;
        end
        transformed_image(round(new_y),round(new_x)) = value;
    end
end
%Plot the original image 1 and transformed image 1
figure; 
%subplot(1,2,1); 
%imshow(original_image2);
%subplot(1,2,2); 
imshow(transformed_image,[]);


%Plotting the two images side by side with (a line connecting) the original T points in image1 and transformed T points over image2
%{
figure(5);
both = cat(2,original_image1,original_image2);
imshow(both,[]);
h1 = vl_plotframe(f) ;
h2 = vl_plotframe(f) ;
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;

f_transform = f;
lines = [];
for i = 1:size(f_transform,2)
    column = f_transform(:,i);
    x = column(1);
    y = column(2);
    A_pixel = [x,y, 0,0,1,0;0,0,x,y,0,1];
    b_pixel = A_pixel * best_parameters_x;
    new_x = round(b_pixel(1));
    new_y = round(b_pixel(2));
    if new_x < 1
       new_x = 1;
    end
    if new_y < 1
       new_y = 1;
    end
    new_x = new_x + size(original_image1,2);
    f_transform(1,i) = new_x; 
    f_transform(2,i) = new_y;
    lines = [lines;x,new_x,y,new_y];
end

h1 = vl_plotframe(f_transform) ;
h2 = vl_plotframe(f_transform) ;
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;
    
figure(6);
both = cat(2,original_image1,original_image2);
imshow(both,[]);
h1 = vl_plotframe(f) ;
h2 = vl_plotframe(f) ;
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ; 
    
h1 = vl_plotframe(f_transform) ;
h2 = vl_plotframe(f_transform) ;
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;

orange = [0.8500, 0.3250, 0.0980];
light_blue = [0.3010, 0.7450, 0.9330];
colors = {'r','g','b','y','m','c','w','k',orange,light_blue};

for row = 1:size(lines,1)
    line_coordinates = lines(row,:);
    x = line_coordinates(1);
    new_x = line_coordinates(2);
    y = line_coordinates(3);
    new_y = line_coordinates(4);
    %index = mod(row,10) + 1;
    %line([x new_x],[y new_y],'Color',colors{index} ,'LineWidth',2);
    line([x new_x],[y new_y],'Color','b' ,'LineWidth',2);
end
%}    


end

