
function RANSAC(image1,image2,matches,p_amount)

perm_matches = randperm(size(matches,2));
pair_index = perm_matches(1:p_amount);
pair_indices = matches(:,pair_index);
index_points_im1 = pair_indices(1,:);
indeX_points_im2 = pair_indices(2,:);

%f1_info = f(:,index_points_im1);
%f2_info = f(:,index_points_im2);
%keyboard


end