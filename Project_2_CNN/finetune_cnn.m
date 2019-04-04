function [net, info, expdir] = finetune_cnn(varargin)

%% Define options
run(fullfile('D:', 'Program Files' , 'MatConvNet', 'matlab', 'vl_setupnn.m'));

opts.modelType = 'lenet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile('data', ...
  sprintf('cnn_assignment-%s', opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = './data/' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb-stl.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.networkType = 'simplenn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

opts.train.gpus = [0];



%% update model

net = update_model();

%% TODO: Implement getIMDB function below

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getIMDB() ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

%%
net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

trainfn = @cnn_train ;
[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 2)) ;

expdir = opts.expDir;
end
% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

end

function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end

end

% -------------------------------------------------------------------------
function imdb = getIMDB()
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
classes = {'airplanes', 'birds', 'ships', 'horses', 'cars'};
splits = {'train', 'test'};

%% TODO: Implement your loop here, to create the data structure described in the assignment
%% Use train.mat and test.mat we provided from STL-10 to fill in necessary data members for training below
%% You will need to, in a loop function,  1) read the image, 2) resize the image to (32,32,3), 3) read the label of that image

%%% Our code
% Load data into memory
train= load("train");
x_train = train.X;
y_train = train.y;
test = load("test");
x_test = test.X;
y_test = test.y;

% Initialize counters and arrays of indices
train_size = 0;
test_size = 0;
indices_train = [];
indices_test = [];

% First loop over the data to find the images that will be used
for i = 1:size(x_train, 1)
    train_size = train_size + 1;
    %Check if the image is of a class that we are interested in
    if y_train(i) == 1
        y_train(i) = 1;
    elseif y_train(i) == 2
        y_train(i) = 2;
    elseif y_train(i) == 9
        y_train(i) = 3;
    elseif y_train(i) == 7
        y_train(i) = 4;
    elseif y_train(i) == 10
        y_train(i) = 5;
    else
        train_size = train_size - 1;
        continue
    end
    %Add to list of indices
    indices_train = [indices_train; i];
end

for i = 1:size(x_test, 1)
    test_size = test_size + 1;
    %Check if the image is of a class that we are interested in
    if y_test(i) == 1
        y_test(i) = 1;
    elseif y_test(i) == 2
        y_test(i) = 2;
    elseif y_test(i) == 9
        y_test(i) = 3;
    elseif y_test(i) == 7
        y_test(i) = 4;
    elseif y_test(i) == 10
        y_test(i) = 5;
    else
        test_size = test_size - 1;
        continue
    end
    %Add to list of indices
    indices_test = [indices_test; i];
end

% Initialize variables
total_size = train_size + test_size;
im_data = zeros(32, 32, 3, total_size);
labels = zeros(total_size, 1);
sets = zeros(total_size, 1);
im_size = 96;

% Loop over the images that we picked above
for i = 1:train_size
    % i is used as an index for both the indices_train and the
    % im_data/labels/sets arrays. j is the index of the image in the
    % original x_train array
    j = indices_train(i);
    I = x_train(j,:); 
    
    % Reshape into three channels and resize
    channel_1 = reshape( I(1:im_size*im_size),im_size,im_size);
    channel_2 = reshape( I(im_size*im_size+1:im_size*im_size*2),im_size,im_size);
    channel_3 = reshape( I(im_size*im_size*2+1:end),im_size,im_size);
    I = cat(3,cat(3,channel_1,channel_2),channel_3);
    I = imresize(I, [32,32]);
    
    % Assign values to the data, labels and sets according to the current image
    im_data(:,:,:,i) = I(:,:,:);
    labels(i) = y_train(j);
    sets(i) = 1;
end

for i = 1:test_size
    % i is used as an index for the indices_train array
    %j is the index of the image in the original x_test array
    j = indices_test(i);
    %k is the index to be used in the im_data/labels/sets arrays
    k = i + train_size;
    I = x_test(j,:); 
    
    % Reshape into three channels and resize
    channel_1 = reshape( I(1:im_size*im_size),im_size,im_size);
    channel_2 = reshape( I(im_size*im_size+1:im_size*im_size*2),im_size,im_size);
    channel_3 = reshape( I(im_size*im_size*2+1:end),im_size,im_size);
    I = cat(3,cat(3,channel_1,channel_2),channel_3);
    I = imresize(I, [32,32]);
    
    % Assign values to the data, labels and sets according to the current image
    im_data(:,:,:,k) = I(:,:,:);
    labels(k) = y_test(j);
    sets(k) = 2;
end

%%% end our code

%%
% subtract mean
dataMean = mean(im_data(:, :, :, sets == 1), 4);
im_data = bsxfun(@minus, im_data, dataMean);

imdb.images.data = im_data ;
imdb.images.labels = single(labels) ;
imdb.images.set = sets;
imdb.meta.sets = {'train', 'val'} ;
imdb.meta.classes = classes;

perm = randperm(numel(imdb.images.labels));
imdb.images.data = imdb.images.data(:,:,:, perm);
imdb.images.labels = imdb.images.labels(perm);
imdb.images.set = imdb.images.set(perm);

end
