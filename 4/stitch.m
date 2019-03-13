function stitched_image = stitch(left, right)

matches = keypoint_matching(right, left);

amount_matches = 10;
n_repeat = 20;
[m1, m2, m3, m4, t1, t2] = RANSAC(right,left,matches,amount_matches,n_repeat);
parameters_x = [m1; m2; m3; m4; t1; t2];

%Transform the image so that the stitching is possible
transformed_image = zeros(size(right));
for y = 1:size(right,1)
    for x = 1: size(right,2)
        value = right(y,x);
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
        transformed_image(round(new_y),round(new_x)) = value;
        
        %Find transformed corner coordinates
        if y == 1
            if x == 1
                cor_1 = [round(new_y),round(new_x)];
            elseif x == size(right,2)
                cor_2 = [round(new_y),round(new_x)];
            end
            
        end
        
        if y == size(right,1)
            if x == 1
                cor_3 = [round(new_y),round(new_x)];
            elseif x == size(right,2)
                cor_4 = [round(new_y),round(new_x)];
            end
            
        end
    end
end

figure; imshow(transformed_image,[]);

%Find the actual minimum size necessary to display the transformed image
transformed_size = [max(max(cor_1(1), cor_2(1)), max(cor_3(1), cor_4(1)))- min(min(cor_1(1), cor_2(1)), min(cor_3(1), cor_4(1))), 
    max(max(cor_1(2), cor_2(2)), max(cor_3(2), cor_4(2)))-min(min(cor_1(2), cor_2(2)), min(cor_3(2), cor_4(2)))];

%Get the sift features again
[f,d] = vl_sift(left);
[f2,d2] = vl_sift(single(transformed_image));
[key_matches, ~] = vl_ubcmatch(d, d2);

offset_y = zeros(size(key_matches, 2), 1);
offset_x = zeros(size(key_matches, 2), 1);

%For each match, find the offset of their coordinates in the images
for i = 1:size(key_matches, 2)
   l = key_matches(1, i);
   r = key_matches(2, i);
   offset_y(i) = round(f(1,l) - f2(1,r));
   offset_x(i) = round(f(2,l) - f2(2,r));
end

%Find the most common offset of the matched sift features
most_common_offset_y = mode(offset_y);
most_common_offset_x = mode(offset_x);

%Create matrix of necessary size
stitched_image = zeros(max(size(left, 1), transformed_size(1)), max(size(left, 2), transformed_size(2) + most_common_offset_x));


if size(stitched_image, 1) == size(left, 1)
    %Place left image on the left
    for y = 1:size(left, 1)
        for x = 1:size(left, 2)
            stitched_image(y,x) = left(y,x);
        end
    end
    %Place transformed right image with the most common offset of matching
    %sift features
    for y = 1:size(transformed_image, 1)
        for x = 1:size(transformed_image, 2)
            if transformed_image(y,x) ~= 0
               stitched_image(y+most_common_offset_y,x+most_common_offset_x) = transformed_image(y,x); 
            end
        end
    end
end

end 