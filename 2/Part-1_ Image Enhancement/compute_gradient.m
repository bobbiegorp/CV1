function [Gx, Gy, im_magnitude,im_direction] = compute_gradient(image)
% Get image
img = im2double(imread(image));

% Define sobel matrices
sobel_x = double([1 0 -1; 2 0 -2; 1 0 -1]);
sobel_y = double([1  2 1; 0 0 0; -1 -2 -1]);

% Calculate results
Gx = conv2(sobel_x,img);
Gy = conv2(sobel_y,img);
im_magnitude = sqrt(Gx.^2 + Gy.^2);
im_direction = atan2(-Gy,Gx).*(180/pi);

% Plot results
figure, imshow(Gx,[]), title("Gradient in x-direction")
figure, imshow(Gy, []), title("Gradient in y-direction")
figure, imshow(im_magnitude, []), title("Gradient magnitude")
figure, imshow(im_direction, []), title("Gradient direction")

end

