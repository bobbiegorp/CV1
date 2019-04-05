

function demo_bow()

close all
clear all
%run("./vlfeat-0.9.21/toolbox/vl_setup.m");
%run("/home/marvin/Documenten/Master_AI_leerjaar_1/Computer vision 1/vlfeat-0.9.21/toolbox/vl_setup.m");

%Setting seed
rng(1);

train= load("train");
class_names = train.class_names;
x_train = train.X;
y_train = train.y;

test = load("test");
x_test = test.X;
y_test = test.y;

color_space = 1; %1 is gray, 2 is RGB, 3 is opponent
sift_method = 1;  
amount_clusters = 400;
target_class_index = 1; %1,2,3,7,9,  ["airplane","bird", "car", "horse","ship", ]

amount_for_v = 150; %Amount for constructing vocabulary
amount_to_test = 800;

BoWClassifier(x_train,y_train,x_test,y_test,class_names,color_space,sift_method,amount_clusters,amount_for_v, amount_to_test,target_class_index);


end