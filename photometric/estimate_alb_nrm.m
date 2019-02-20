function [ albedo, normal ] = estimate_alb_nrm( image_stack, scriptV, shadow_trick)
%COMPUTE_SURFACE_GRADIENT compute the gradient of the surface
%   image_stack : the images of the desired surface stacked up on the 3rd
%   dimension
%   scriptV : matrix V (in the algorithm) of source and camera information
%   shadow_trick: (true/false) whether or not to use shadow trick in solving
%   	linear equations
%   albedo : the surface albedo
%   normal : the surface normal

[h, w, ~] = size(image_stack);
if nargin == 2
    shadow_trick = true;
end

% create arrays for 
%   albedo (1 channel)
%   normal (3 channels)
albedo = zeros(h, w, 1);
normal = zeros(h, w, 3);

% =========================================================================
% YOUR CODE GOES HERE
% for each point in the image array
%   stack image values into a vector i
%   construct the diagonal matrix scriptI
%   solve scriptI * scriptV * g = scriptI * i to obtain g for this point
%   albedo at this point is |g|
%   normal at this point is g / |g|


warning off
%shadow_trick = false;
%Q1

%Variable stop to indicate how many images to use
stop = size(image_stack,3);
%size(image_stack)
%stop = 121;
%scriptV = scriptV(1:stop,:);
% for each point in the image array
for x = 1:w
    for y = 1:h
        %stack image values into a vector i
        %disp(size(image_stack(y,x,1)))
        i = image_stack(y,x,1:stop);
        %construct the diagonal matrix scriptI
        i = squeeze(i);
        if shadow_trick
            scriptI = diag(i);
            %solve scriptI * scriptV * g = scriptI * i to obtain g for this point
            g = linsolve(scriptI*scriptV,scriptI*i);
        else
            g = linsolve(scriptV,i);
        end
        albedo(y,x) = norm(g);
        normal(y,x,:) = g./albedo(y,x);
    end
end
%imwrite(albedo,"./Q1_images/25_albedo_25_nst.jpg")
%imwrite(normal,"./Q1_images/25_normal_25_nst.jpg")
%figure
%imshow(albedo)
%figure
%imshow(normal)

%Q2, same code as Q1, but with incremental loop
%{
for increment=[2:25]
    stop = increment;
    scriptv = scriptV(1:stop,:);
    % for each point in the image array
    for x = 1:w
        for y = 1:h
            %stack image values into a vector i
            i = image_stack(x,y,1:stop);
            %construct the diagonal matrix scriptI
            i = squeeze(i);
            scriptI = diag(i);
            %solve scriptI * scriptV * g = scriptI * i to obtain g for this point
            g = linsolve(scriptI*scriptv,scriptI*i);
            albedo(x,y) = norm(g);
            normal(x,y,:) = g./albedo(x,y);
        end
    end
    imwrite(albedo,"./Q1_images/25_albedo_" + string(stop) + ".jpg")
    imwrite(normal,"./Q1_images/25_normal_" + string(stop) + ".jpg")
%}

% =========================================================================

end

