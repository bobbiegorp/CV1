function [ p, q, SE ] = check_integrability( normals )
%CHECK_INTEGRABILITY check the surface gradient is acceptable
%   normals: normal image
%   p : df / dx
%   q : df / dy
%   SE : Squared Errors of the 2 second derivatives

% initalization
[h, w, ~] = size(normals);
p = zeros(h,w);
q = zeros(h,w);
%p = zeros(size(normals));
%q = zeros(size(normals));
%SE = zeros(size(normals));
SE = zeros(h,w);

second_deriv_xy = zeros(h,w);%size(normals));
second_deriv_yx = zeros(h,w);%size(normals));
% ========================================================================
% YOUR CODE GOES HERE
% Compute p and q, where
% p measures value of df / dx
% q measures value of df / dy
for x = 1:w
    for y = 1:h
        normal_point = normals(y,x,:);
        n = squeeze(normal_point);
        p_value = n(1)/n(3);
        q_value = n(2)/n(3);
        p(y,x) = p_value;
        q(y,x) = q_value;
    end
end
% ========================================================================


p(isnan(p)) = 0;
q(isnan(q)) = 0;


% ========================================================================
% YOUR CODE GOES HERE
% approximate second derivate by neighbor difference
% and compute the Squared Errors SE of the 2 second derivatives SE

for x = 1:w
    for y = 1:h
        %p = p(x,y);
        %q = q(x,y);
        %second_xy = n(2)/p;
        %second_yx = n(1)/q;
        %diff = (second_xy - second_yx)^2;
        %SE(x,y) = diff;

        if x > 1 
            second_deriv_xy(y,x) = p(y,x) - p(y,x-1);
        else
            second_deriv_xy(y,x) = 0;
        end 
        
        if y > 1
            second_deriv_yx(y,x) = q(y,x) - q(y-1,x);
        else
            second_deriv_yx(y,x) = 0;
        end
        
        diff = (second_deriv_xy(y,x) - second_deriv_yx(y,x))^2;
        SE(y,x) = diff;
    end
end

%figure(1)
%imshow(second_deriv_xy)
%figure(2)
%imshow(second_deriv_yx)
%xy_edges = second_deriv_xy + second_deriv_yx;
%figure(3)
%imshow(xy_edges)



% ========================================================================




end

