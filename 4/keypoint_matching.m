

function key_matches = keypoint_matching(original_image1, original_image2)

if size(original_image1,3) == 3
    disp("hallo")
    image1 = single(rgb2gray(original_image1));
    image2 = single(rgb2gray(original_image2));
else
    image1 = single(original_image1);
    image2 = single(original_image2);
end 

[f,d] = vl_sift(image1);
[f2,d2] = vl_sift(image2);
[key_matches, scores] = vl_ubcmatch(d, d2) ;

%For self illustration purposes image 1 
figure; imshow(original_image1);
perm = randperm(size(f,2)) ;
sel = perm(1:50) ;
h1 = vl_plotframe(f(:,sel)) ;
h2 = vl_plotframe(f(:,sel)) ;
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;
%h3 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;
%set(h3,'color','g') ;

%For self illustration purposes image 2
figure; imshow(original_image2);
perm = randperm(size(f2,2)) ;
sel2 = perm(1:50) ;
h3 = vl_plotframe(f2(:,sel2)) ;
h4 = vl_plotframe(f2(:,sel2)) ;
set(h3,'color','k','linewidth',3) ;
set(h4,'color','y','linewidth',2) ;
%h3 = vl_plotsiftdescriptor(d2(:,sel),f2(:,sel)) ;
%set(h3,'color','g') ;


% For self illustration purposes, Both images next to each other 
both = cat(2,original_image1,original_image2);
figure; imshow(both);
h1 = vl_plotframe(f(:,sel)) ;
h2 = vl_plotframe(f(:,sel)) ;
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;

%For self illustration purposes, adjust the frames of second picture to width of first image
frames_2 = f2(:,sel2);
frames_2(1,:) = frames_2(1,:) + size(image1,2);
h3 = vl_plotframe(frames_2) ;
h4 = vl_plotframe(frames_2) ;
set(h3,'color','k','linewidth',3) ;
set(h4,'color','y','linewidth',2) ;


% Both images next to each other with the 10 pairs, for assignment
figure; imshow(both);

% Pick 10 random matches to draw a line
perm_matches = randperm(size(key_matches,2));
pair_index = perm_matches(1:10);
line_pair_indices = key_matches(:,pair_index);
%h1 = vl_plotframe(f(:,sel)) ;
%h2 = vl_plotframe(f(:,sel)) ;
%set(h1,'color','k','linewidth',3) ;
%set(h2,'color','y','linewidth',2) ;

for pair = line_pair_indices
    index_image1 = pair(1);
    index_image2 = pair(2);

    f1_info = f(:,index_image1);
    x = f1_info(1);
    y = f1_info(2);

    f2_info =  f2(:,index_image2);
    f2_info(1) = f2_info(1) + size(image1,2);
    x2 = f2_info(1);
    y2 = f2_info(2); 
    
    line([x x2],[y y2],'Color','r','LineWidth',2);
    
    %Plot the frames of 1 and 2
    h1 = vl_plotframe(f1_info) ;
    h2 = vl_plotframe(f1_info) ;
    set(h1,'color','k','linewidth',3) ;
    set(h2,'color','y','linewidth',2) ;
    
    h1 = vl_plotframe(f2_info) ;
    h2 = vl_plotframe(f2_info) ;
    set(h1,'color','k','linewidth',3) ;
    set(h2,'color','y','linewidth',2) ;
    
end 

end