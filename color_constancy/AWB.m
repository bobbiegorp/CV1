I=imread("awb.jpg");
GrayWorld(I);

function [GW_image] = GrayWorld(I)
    % Read image and specify size of output image
    [Row Col Layer] = size(I); 
    GW_image = uint8(zeros(Row, Col, Layer));

    % Retrieve the RGB values
    R = I(:,:,1);
    G = I(:,:,2);
    B = I(:,:,3);

    % Calculate means of R,G and B values
    R_mean = mean(mean(R));
    G_mean = mean(mean(G));
    B_mean = mean(mean(B));

    % Compute overall average
    Average = mean([R_mean, G_mean, B_mean]);

    % Retrieve scaling factor
    R_scale = Average / R_mean;
    G_scale = Average / G_mean;
    B_scale = Average / B_mean;

    % Scale image
    GW_image(:,:,1) = R * R_scale;
    GW_image(:,:,2) = G * G_scale;
    GW_image(:,:,3) = B * B_scale;

    % Output resulted image and original image
    subplot(1,2,1)
    imshow(GW_image)
    title("Corrected Image");
    subplot(1,2,2)
    imshow(I)
    title("Original Image");  
end