function [ imOut ] = denoise( image, kernel_type, varargin)
img = imread(image);
img = double(img);

switch kernel_type
    case 'box'      
        box1 = uint8(imboxfilt(img,varargin{1}(1)));
        box2 = uint8(imboxfilt(img,varargin{1}(2)));
        box3 = uint8(imboxfilt(img,varargin{1}(3)));
       
        %Plot boxfilter images for Question 7.1
        figure(1)
        subplot(1,4,1), imshow(uint8(img)), title('Original');
        subplot(1,4,2), imshow(box1), title('3x3');
        subplot(1,4,3), imshow(box2), title('5x5');
        subplot(1,4,4), imshow(box3), title('7x7');
        suptitle('Box Filtering: image1 saltpepper');
           
        %Q7.2 Calculate the PSNR of the box filtered image with the original
        %image 
        PSNR = [myPSNR(img, box1), myPSNR(img, box2), myPSNR(img, box3)]
        
    case 'median'
        med1 = uint8(medfilt2(img ,[varargin{1}(1) varargin{1}(1)]));
        med2 = uint8(medfilt2(img ,[varargin{1}(2) varargin{1}(1)]));
        med3 = uint8(medfilt2(img ,[varargin{1}(3) varargin{1}(1)]));
        
        %Plot median filter images for Question 7.1
        figure(1)
        subplot(1,4,1), imshow(uint8(img)), title('Original Image');
        subplot(1,4,2), imshow(med1), title('3x3');
        subplot(1,4,3), imshow(med2), title('5x5');
        subplot(1,4,4), imshow(med3), title('7x7');
        suptitle('Median filtering: image1 gaussian ');
        
        %Q7.2 Calculate the PSNR of the median filtered image with the original
        %image
        PSNR_Median = [myPSNR(img, med1), myPSNR(img, med2), myPSNR(img, med3)]
        
    case 'gaussian'
        img = imread('image1_gaussian.jpg');
        img_double = im2double(img);
        figure(1), suptitle('Examples of gaussian filter with sigma ranging from 0.125 to 1.25');
        hold on

%Q7.4 Calculating PSNR and creating figures for both different kernel sizes as
%different standard deviations:
        for i = 1:10
             h = gauss2D(i/8,3);
             i/8
             I=imfilter(img_double,h); 
             subplot(2,5,(i));
             imshow(I), xlabel(num2str(i/8));
             PSNR = myPSNR('image1.jpg',uint8(I));
        end
    
        figure(2), suptitle('Examples of gaussian filter with kernel size ranging from 3 to 7');
        hold on
        j = 1; %Initializing j, to subplot the images
        for i = 3:2:7
            h = gauss2D(0.75,i);
            I=imfilter(img_double,h);        
            subplot(1,3,j);
            imshow(I), xlabel(num2str(i));
            j = j + 1;
        end    
   
%Question 7.5 , with manipulation of the standard deviation (sigma). 
        for i = 0.1:0.1:1
             
             h = gauss2D(i,3);
             I=imfilter(img_double,h);
             PSNR = myPSNR('image1.jpg', uint8(I));

        end
    end
end

