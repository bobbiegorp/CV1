number = 1;

%Theta 
%{
for theta_value = [0,pi/3,2*pi/4,2*pi/3,2*pi/2]
    myGabor = createGabor(3,theta_value,15,pi/2,0.4);
    myGabor_real = myGabor(:,:,1); 
    myGabor_imaginary = myGabor(:,:,2);
    
    subplot(2,5,number), imshow(myGabor_real,[]);
    title("Real - Theta =  " + num2str(theta_value) );
    subplot(2,5,number+5), imshow(myGabor_imaginary, []);
    title("Imaginary - Theta = " + num2str(theta_value));
    number = number +1;
end
%}

%{
%sigma
for sigma_value = [2,4,6,8,10]
    myGabor = createGabor(sigma_value,pi/3,15,pi/2,0.4);
    myGabor_real = myGabor(:,:,1); 
    myGabor_imaginary = myGabor(:,:,2);
    
    subplot(2,5,number), imshow(myGabor_real,[]);
    title("Real - Sigma =  " + num2str(sigma_value) );
    subplot(2,5,number+5), imshow(myGabor_imaginary, []);
    title("Imaginary - Sigma= " + num2str(sigma_value));
    number = number +1;
end
%}


%sigma
for gamma_value = [0.2,0.4,0.6,0.8,1.0]
    myGabor = createGabor(6,pi/3,15,pi/2,gamma_value);
    myGabor_real = myGabor(:,:,1); 
    myGabor_imaginary = myGabor(:,:,2);
    
    subplot(2,5,number), imshow(myGabor_real,[]);
    title("Real - Gamma =  " + num2str(gamma_value) );
    subplot(2,5,number+5), imshow(myGabor_imaginary, []);
    title("Imaginary - Gamma = " + num2str(gamma_value));
    number = number +1;
end

