function recoloring (image)
img = imread(image);
img_green = img;
%Replacing the first and third columns of the color by 0 of the colour scheme.
img_green(:,:,[1 3]) = 0;

figure(1),
subplot(1,2,1), imshow(img), title('Original ball image');
subplot(1,2,2), imshow(img_green), title('Recolored ball image');  
end