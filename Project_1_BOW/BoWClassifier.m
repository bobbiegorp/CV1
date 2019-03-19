
function BoWClassifier()

run("/home/marvin/Documenten/Master_AI_leerjaar_1/Computer vision 1/vlfeat-0.9.21/toolbox/vl_setup.m");

%You can use VLFeat functions for dense SIFT (e.g. vl dsift) 
%and key points SIFT descriptor extraction (e.g. vl sift).
%Moreover, it is expected that you implement not only for grayscale SIFT, 
%but also for RGB-SIFT and opponent-SIFT.

%For color 2 approaches?
%Turn graysacle, to find keypoints.

train= load("train");
class_names = train.class_names;
x_train = train.X;
y_train = train.y;

color_space = 1; %gray
sift_method = 1;
%Keypoints is method 1
%Dense sampled region is method 2

amount_classes = length(class_names);
total_training = size(x_train,1);

disp(class_names);

%Sort all data so can take subsets of data for each class
%classes = ["airplane","bird", "ship", "horse","car"];
classes = [1,2,3,7,9];
sorted_data = containers.Map('KeyType','single' ,'ValueType','any');
class_lookup = containers.Map('KeyType','single' ,'ValueType','any'); 

%Easier to see if class index is wanted with so many images, so set to 1 or
%0 for lookup and initliaze empty matrix f
for i = 1:amount_classes
   if ismember(i,classes)
       sorted_data(i) = [];
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
        sorted_data(class_index) = [sorted_data(class_index); im_data];
    end
end

%Take subset of each class, extract descriptor for clustering visual words
cluster_data = [];
amount_for_v = 150;
for class_index = classes
    
    class_im_data = sorted_data(class_index);
    subset = class_im_data(1:amount_for_v,:);
    for im_data_index = 1:amount_for_v
        im_data = subset(im_data_index,:); 
        [f,d] = Get_SIFTData(im_data,color_space,sift_method); 
        cluster_data = [cluster_data;d'];
    end
end 

amount_clusters = 400;
[~,centroids] = kmeans(double(cluster_data),amount_clusters);


%Create histogram for reamining training images of all classes
histogram_total = zeros(1,amount_clusters); 
counter = 0;
histogram_data = containers.Map('KeyType','single' ,'ValueType','any');

for class_index = classes
    
    class_im_data = sorted_data(class_index);
    subset = class_im_data(amount_for_v+1:size(class_im_data,1),:);
    histogram_class = zeros(size(subset,1),amount_clusters);
    
    for im_data_index = 1:size(subset,1)
        
        im_data = subset(im_data_index,:); 
        [f,d] = Get_SIFTData(im_data);
        histogram = zeros(1,amount_clusters);
        histogram_sum = size(d,2);
        
        %Using k nearest neighbour
        indices = knnsearch(centroids,double(d'));
        for i = 1:size(indices,1)
            min_index = indices(i,1);
            histogram(1,min_index) = histogram(1,min_index) + 1;
            histogram_total(1,min_index) = histogram_total(1,min_index) + 1;
            counter = counter + 1;
        end
        
        %Normalize histograms
        histogram = histogram/histogram_sum;
        
        %Store the histogram of image
        histogram_class(im_data_index,:) = histogram;
        
        %Use pairwise Euclidecian distance search
        %{
        for descriptor = d
            stack = [descriptor';centroids];
            distances = pdist(double(stack),'euclidean');
            [~,min_index] = min(distances(1:amount_clusters));
            histogram(1,min_index) = histogram(1,min_index) + 1;
            histogram_total(1,min_index) = histogram_total(1,min_index) + 1;
            counter = counter + 1;
        end
        %}
        
        %Use one on one comparison and than min
        %for descriptor = d
        %   norm(descriptor) 
        %end
    end
    
    histogram_data(class_index) = histogram_class;
    %disp(size(histogram_class));
end 

%disp(sum(histogram_total));

%Testing prediction temporarily on training data
histogram_class1 = histogram_data(1);
amount_instances = size(histogram_class1,1);

training_labels = zeros(5*amount_instances,1);
training_labels(1:amount_instances,1) = 1;

training_matrix = zeros(5*amount_instances,amount_clusters);
training_matrix(1:amount_instances,:) = histogram_class1;

histogram_class2 = histogram_data(2);
histogram_class3 = histogram_data(3);
histogram_class7 = histogram_data(7);
histogram_class9 = histogram_data(9);

training_matrix(amount_instances+1:2*amount_instances,:) = histogram_class2;
training_matrix(2*amount_instances+1:3*amount_instances,:) = histogram_class3;
training_matrix(3*amount_instances+1:4*amount_instances,:) = histogram_class7;
training_matrix(4*amount_instances+1:5*amount_instances,:) = histogram_class9;

%model1 = svmtrain(training_labels, training_matrix,[ 'libsvm_options']);
model1 = fitcsvm(training_matrix,training_labels);

%Predicts only negative due to class imbalance. For dense RGB SIFT should
%work better than random guess (MAP of 0.2) as stated on piazza
%Or use less data perhaps. 
[label,score] = predict(model1,histogram_class2);

keyboard
%

end 

function [f,d] =  Get_SIFTData(im_data,color_space,sift_method)

    size_x = 96;
    size_y = 96;

    channel_1 = reshape( im_data(1:size_x*size_y),size_y,size_x);
    channel_2 = reshape( im_data(size_x*size_y+1:size_x*size_y*2),size_y,size_x);
    channel_3 = reshape( im_data(size_x*size_y*2+1:end),size_y,size_x);
    image = cat(3,cat(3,channel_1,channel_2),channel_3);
    [f,d] = vl_sift(im2single(rgb2gray(image))); %128 by x keypoints, take tranpose for stack
    
    %{
    %Plot pointso n image
    figure; imshow(image);
    perm = randperm(size(f,2)) ;
    sel = perm(1:10) ;
    h1 = vl_plotframe(f(:,sel)) ;
    h2 = vl_plotframe(f(:,sel)) ;
    set(h1,'color','k','linewidth',3) ;
    set(h2,'color','y','linewidth',2) ;
    keyboard
    %}
end

