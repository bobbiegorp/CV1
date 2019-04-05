

function BoWClassifier(x_train,y_train,x_test,y_test,class_names,color_space,sift_method,amount_clusters,amount_for_v, amount_to_test,target_class_index)

total_classes = length(class_names);
total_training = size(x_train,1);
total_test = size(y_test,1);

disp(class_names);

%Sort all data so can take subsets of data for each class
%classes = ["airplane","bird", "car", "horse","ship", ]
classes = [1,2,3,7,9];
sorted_train_data = containers.Map('KeyType','single' ,'ValueType','any');
sorted_test_data = containers.Map('KeyType','single' ,'ValueType','any');
class_lookup = containers.Map('KeyType','single' ,'ValueType','any'); 

%Easier to see if class index is wanted with so many images, so set to 1 or
%0 for lookup and initliaze empty matrix for data
for i = 1:total_classes
   if ismember(i,classes)
       sorted_train_data(i) = [];
       sorted_test_data(i) = [];
       class_lookup(i) = 1;
   else
       class_lookup(i) = 0;
   end
end

%Get all the data only relevant to our classes and put them in lookup matrix
for index = 1:total_training
    im_data = x_train(index,:);
    class_index = y_train(index,:);
    if class_lookup(class_index)
        sorted_train_data(class_index) = [sorted_train_data(class_index); im_data];
    end
end
disp("Done sorting data for training set")

for index = 1:total_test
    im_data = x_test(index,:);
    class_index = y_test(index,:);
    if class_lookup(class_index)
        sorted_test_data(class_index) = [sorted_test_data(class_index); im_data];
    end
end
disp("Done sorting data for test set")

%Take subset of each class of training, extract descriptor for clustering visual words
cluster_data = [];
for class_index = classes
    
    class_im_data = sorted_train_data(class_index);
    subset = class_im_data(1:amount_for_v,:);
    for im_data_index = 1:amount_for_v
        im_data = subset(im_data_index,:); 
        [f,d] = Get_SIFTData(im_data,color_space,sift_method); 
        cluster_data = [cluster_data;d'];
    end
end 

[partition,centroids] = kmeans(double(cluster_data),amount_clusters);
disp("Done clustering for visual vocabulary")

%Train
%Create histogram for remaining training images of all classes
amount_to_train = 500 - amount_for_v; %Rest of images not used for clustering
train_end = amount_for_v + amount_to_train;

if train_end > size(class_im_data,1)
    train_end = size(class_im_data,1);
end

histogram_train_data = histogram_quantization(sorted_train_data,amount_for_v+1,train_end,amount_clusters,centroids,classes,color_space,sift_method);
disp("Done with training data quantization")

%Test
%Create histogram for remaining testing images of all classes
%amount_to_test = 800;
histogram_test_data = histogram_quantization(sorted_test_data,1,amount_to_test,amount_clusters,centroids,classes,color_space,sift_method);
disp("Done with test data quantization")

%Combining training data for training, labeled with a 1 for the target
%class
[training_matrix,training_labels] = combine_data(histogram_train_data,amount_to_train,classes,target_class_index,amount_clusters);
disp("Done combining training data for training SVM")

%model1 = svmtrain(training_labels, training_matrix,[ 'libsvm_options']);
%Train SVM model and turn score into class probabilities
model1 = fitcsvm(training_matrix,training_labels,'KernelFunction','rbf');
compact_svm = compact(model1);
svm_model = fitPosterior(compact_svm,training_matrix,training_labels);
disp("Done training SVM")

%Testing on test data, after combining into matrix and labels
[test_matrix,test_labels] = combine_data(histogram_test_data,amount_to_test,classes,target_class_index,amount_clusters);
disp("Done combining test data for testing")

%Predict 
[predict2,class_prob2] = predict(svm_model,test_matrix);
disp(sum(predict2));

[test_acc,map_score,ranked_prob,ranked_indices_test] = computeResults(class_prob2,test_labels,predict2);
disp("Done with predictions, computing score")
disp("Accuracy: " + test_acc);
disp("mAP Score: " + map_score);

%Display top and worst 5 confidence probabiltiies, name of file
if sift_method == 1
    sift_approach = "_keypoints_";
else
    sift_approach = "_dense_";
end

if color_space == 1
    color = "_gray_";
else if color_space == 2
    color = "_RGB_";
else if color_space == 3
    color = "_opponentRGB_";
    end
    end
end

disp(class_names(target_class_index)); 
properties_name = num2str(amount_clusters) + "_" + sift_approach + color + ".png";

%Display 5 best confdence probabilties
[classes_of_display,top5] = display_images(1,ranked_indices_test,predict2,sorted_test_data,amount_to_test,classes,class_names,target_class_index,properties_name);
disp(classes_of_display);

%Display 5 worst confidence probabilities
[classes_of_display,worse5] = display_images(0,ranked_indices_test,predict2,sorted_test_data,amount_to_test,classes,class_names,target_class_index,properties_name);
disp(classes_of_display);

%Display top5 at top and worse5 below it in 1 iamge
combined = stack_images(top5,worse5);
%imwrite(combined ,class_names(target_class_index) + "_combined_" + properties_name);

%keyboard
end 

function [f,d] =  Get_SIFTData(im_data,color_space,sift_method)

    size_x = 96;
    size_y = 96;

    channel_1 = reshape( im_data(1:size_x*size_y),size_y,size_x);
    channel_2 = reshape( im_data(size_x*size_y+1:size_x*size_y*2),size_y,size_x);
    channel_3 = reshape( im_data(size_x*size_y*2+1:end),size_y,size_x);
    image = cat(3,cat(3,channel_1,channel_2),channel_3);
    
    if sift_method == 1 %Key points
        switch color_space
            %gray
            case 1 %gray
                [f,d] = vl_sift(im2single(rgb2gray(image))); 
            case 2 
                [f,d_full] = vl_sift(im2single(rgb2gray(image))); %Find generic keypoints first
                [f1,d1] = vl_sift(im2single(channel_1),'frames',f); %Channel 1
                [f2,d2] = vl_sift(im2single(channel_2),'frames',f); %Channel 1
                [f3,d3] = vl_sift(im2single(channel_3),'frames',f); %Channel 1
                
                d = cat(2,cat(2,d1,d2),d3);
            case 3
                
                [f,d_full] = vl_sift(im2single(rgb2gray(image))); %Find generic keypoints first
                
                %Transform to opponent space
                opponent_channel_1 = (channel_1 - channel_2)./sqrt(2);
                opponent_channel_2 = (channel_1 + channel_2 - 2*channel_3)./sqrt(6);
                opponent_channel_3 = (channel_1 + channel_2 + channel_3)./sqrt(3);

                [f1,d1] = vl_sift(im2single(opponent_channel_1),'frames',f); %Channel 1
                [f2,d2] = vl_sift(im2single(opponent_channel_2),'frames',f); %Channel 2
                [f3,d3] = vl_sift(im2single(opponent_channel_3),'frames',f); %Channel 3
                
                d = cat(2,cat(2,d1,d2),d3);

        end
        
    else  %Dense sampling
        switch color_space
            %gray
            case 1
                binSize = 8 ;
                magnif = 3 ;
                Is = vl_imsmooth(im2single(rgb2gray(image)), sqrt((binSize/magnif)^2 - .25)) ;
                [f,d] = vl_dsift(Is,'step',10,'size', binSize); 
                %[f,d] = vl_dsift(im2single(rgb2gray(image)),'step',10); %originele manier
            case 2
                binSize = 8 ;
                magnif = 3 ;
                Is = vl_imsmooth(im2single(rgb2gray(image)), sqrt((binSize/magnif)^2 - .25)) ;
                [f,d] = vl_dsift(Is,'step',10,'size', binSize); 
                f(3,:) = binSize/magnif ;
                f(4,:) = 0 ;
                
                [f1, d1] = vl_sift(im2single(channel_1), 'frames', f) ;
                [f2, d2] = vl_sift(im2single(channel_2), 'frames', f) ;
                [f3, d3] = vl_sift(im2single(channel_3), 'frames', f) ;
                d = cat(2,cat(2,d1,d2),d3);   

            case 3
                binSize = 8 ;
                magnif = 3 ;
                Is = vl_imsmooth(im2single(rgb2gray(image)), sqrt((binSize/magnif)^2 - .25)) ;
                [f,d] = vl_dsift(Is,'step',10,'size', binSize); 
                f(3,:) = binSize/magnif ;
                f(4,:) = 0 ;
                
                %Transform to opponent space
                opponent_channel_1 = (channel_1 - channel_2)./sqrt(2);
                opponent_channel_2 = (channel_1 + channel_2 - 2*channel_3)./sqrt(6);
                opponent_channel_3 = (channel_1 + channel_2 + channel_3)./sqrt(3);

                [f1, d1] = vl_sift(im2single(opponent_channel_1), 'frames', f) ;
                [f2, d2] = vl_sift(im2single(opponent_channel_2), 'frames', f) ;
                [f3, d3] = vl_sift(im2single(opponent_channel_3), 'frames', f) ;
                d = cat(2,cat(2,d1,d2),d3);   
                
        end
    end
end


function normalized_histogram = ComputeHistogram(im_data,centroids,amount_clusters,color_space,sift_method)

        [f,d] = Get_SIFTData(im_data,color_space,sift_method);
        histogram = zeros(1,amount_clusters);
        histogram_sum = size(d,2);
        
        %Using k nearest neighbour
        indices = knnsearch(centroids,double(d'));
        for i = 1:size(indices,1)
            min_index = indices(i,1);
            histogram(1,min_index) = histogram(1,min_index) + 1;
        end
        
        %Normalize histograms
        normalized_histogram = histogram/histogram_sum;

end

function histogram_data = histogram_quantization(sorted_data,start_subset,end_subset,amount_clusters,centroids,classes,color_space,sift_method)

    histogram_data = containers.Map('KeyType','single' ,'ValueType','any');

    for class_index = classes

        class_im_data = sorted_data(class_index);

        subset = class_im_data(start_subset:end_subset,:);
        histogram_class = zeros(size(subset,1),amount_clusters);

        for im_data_index = 1:size(subset,1)

            im_data = subset(im_data_index,:); 

            %Get normalized histograms
            histogram = ComputeHistogram(im_data,centroids,amount_clusters,color_space,sift_method);

            %Store the histogram of image in respective class
            histogram_class(im_data_index,:) = histogram;
        end

        histogram_data(class_index) = histogram_class;
        %disp(size(histogram_class));
    end 
  
end

function [accuracy,map_score,ranked_prob,indices_test] = computeResults(class_prob,test_labels,predictions)
    map_sum = 0;
    amount_target = 1;
    correct = 0;
    [ranked_prob,indices_test] = sort(class_prob(:,2),'descend');
    
    for i = 1:length(indices_test)
        index = indices_test(i);
        label = test_labels(index);
        if label
             map_sum = map_sum + amount_target/i;
             amount_target = amount_target + 1;
        end
        
        if label == predictions(index)
            correct = correct + 1;
        end
        
    end
    
    accuracy = correct / size(test_labels,1);
    map_score = map_sum/800;
    
end

function [feature_matrix,target_labels] = combine_data(histogram_features,amount_instances,classes,target_class_index,amount_clusters)

    amount_classes = length(classes);
    feature_matrix = zeros(amount_classes*amount_instances,amount_clusters);
    target_labels = zeros(amount_classes*amount_instances,1);
    
    for i = 0:length(classes)-1
        class_index = classes(i+1);
        histogram_class = histogram_features(class_index);
        
        start_row = i * amount_instances + 1;
        end_row = (i+1) * amount_instances;
        feature_matrix(start_row:end_row,:) = histogram_class;
        if class_index == target_class_index
           target_labels(start_row:end_row,1) = 1; 
        end
    end
   
end

function [classes_of_display,full] = display_images(top_or_bottom_5,ranked_indices_test,predictions,sorted_test_data,amount_to_test,classes,class_names,target_class_index,p_name)
    
    %Take indices where the classifier said is target class
    ranked_index_target_predic = [];
    for i = 1:size(ranked_indices_test,1)
       index = ranked_indices_test(i);
       if predictions(index)
           ranked_index_target_predic = [ranked_index_target_predic index];
       end
    end
    

    if top_or_bottom_5
        indices = ranked_index_target_predic(1:5);
        title_string = "_Top_5_images_";
    else
        indices = ranked_index_target_predic(end:-1:end-4);
        title_string = "_Worst_5_images_";
    end
    
    %Plot them
    size_x = 96;
    size_y = 96;
    classes_of_display = [];
    image_number = 1;
    figure;
    
    dim1 = [];
    dim2 = [];
    dim3 = [];
    white_space = ones(96,25) * 255;

    for rank_index = indices
        index_in_classes = ceil(rank_index/amount_to_test);
        %disp(index_in_classes);
        class_index = classes(index_in_classes);
        classes_of_display = [classes_of_display class_names(class_index)];
        class_data  = sorted_test_data(class_index);
        index_for_data = mod(rank_index,amount_to_test);
        im_data = class_data(index_for_data,:);

        channel_1 = reshape( im_data(1:size_x*size_y),size_y,size_x);
        channel_2 = reshape( im_data(size_x*size_y+1:size_x*size_y*2),size_y,size_x);
        channel_3 = reshape( im_data(size_x*size_y*2+1:end),size_y,size_x);
        image = cat(3,cat(3,channel_1,channel_2),channel_3);

        dim1 = cat(2,dim1,image(:,:,1));
        dim2 = cat(2,dim2,image(:,:,2));
        dim3 = cat(2,dim3,image(:,:,3));
        if image_number < 5
            dim1 = cat(2,dim1,white_space);
            dim2 = cat(2,dim2,white_space);
            dim3 = cat(2,dim3,white_space);
        end

        image_number = image_number + 1;
    end
    full = cat(3,cat(3,dim1,dim2),dim3);
    imshow(full);
    %imwrite(full,class_names(target_class_index) + title_string + p_name);

end

function full = stack_images(top5,worse5)
    
    white_space = ones(42,size(top5,2)) * 255;

    dim1 = cat(1,top5(:,:,1),white_space);
    dim2 = cat(1,top5(:,:,2),white_space);
    dim3 = cat(1,top5(:,:,3),white_space);

    dim1 = cat(1,dim1,worse5(:,:,1));
    dim2 = cat(1,dim2,worse5(:,:,2));
    dim3 = cat(1,dim3,worse5(:,:,3));
    full = cat(3,cat(3,dim1,dim2),dim3);
    figure; imshow(full);
end
