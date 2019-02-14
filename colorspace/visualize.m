function visualize(input_image)
if size(input_image, 3) == 4
    % then it's the four different methods from the grayscale function
    figure('Position', [200 100 840 630]);
    subplot(2,2,1),imshow(input_image(:,:,1)), title('Lightness method');
    subplot(2,2,2),imshow(input_image(:,:,2)), title('Average method');
    subplot(2,2,3),imshow(input_image(:,:,3)), title('Luminosity method');
    subplot(2,2,4),imshow(input_image(:,:,4)), title('Matlab default');
else
    % it's any of the other color space conversions
    figure('Position', [200 100 840 630]);
    subplot(2,2,1),imshow(input_image), title('Image');
    subplot(2,2,2),imshow(input_image(:,:,1)), title('Channel 1');
    subplot(2,2,3),imshow(input_image(:,:,2)), title('Channel 2');
    subplot(2,2,4),imshow(input_image(:,:,3)), title('Channel 3');
end
end

