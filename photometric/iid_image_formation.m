function [ result  ] = iid_image_formation(directory)
    ball = imread('ball.png');
    albedo = imread('ball_albedo.png');
    shading = imread('ball_shading.png');    
    
    %To do correct elementwise multiplication: casting albedo and shading to
    %doubles and divide the values by 255 (max grayscale value) to get double values between 0
    %and 1
    albedo = double(albedo);
    shading = double(shading);
    shading = shading /255;
    %Elementwise multiplication
    final_img = albedo .* shading;
    %Casting back to data type (class) uint8
    final_img = uint8(final_img);

    %Plotting the original, intrinsic and reconstructed images 
    figure(1)
    subplot(1,4,1), imshow(ball), title('Original ball image'), xlabel('A');
    subplot(1,4,2), imshow(albedo), title('Intrinsic image: Albedo')xlabel('B');
    subplot(1,4,3), imshow(shading), title('Intrinsic image: Shading'), xlabel('C');
    subplot(1,4,4), imshow(final_img), title('Reconstructed image'), xlabel('D');
end