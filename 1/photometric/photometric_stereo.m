close all
clear all
clc
 
disp('Part 1: Photometric Stereo')

% obtain many images in a fixed view under different illumination
disp('Loading images...')
%image_dir = './SphereGray5/';   % TODO: get the path of the script
%image_dir = './SphereGray25/';
image_dir = "./MonkeyGray/";
%image_dir = "./SphereColor/";
%image_dir = "./MonkeyColor/";
%image_ext = '*.png';

[image_stack, scriptV] = load_syn_images(image_dir);
[h, w, n] = size(image_stack);
fprintf('Finish loading %d images.\n\n', n);

% compute the surface gradient from the stack of imgs and light source mat
disp('Computing surface albedo and normal map...')
[albedo, normals] = estimate_alb_nrm(image_stack, scriptV);

%For three channels
%[image_stack_2, scriptV] = load_syn_images(image_dir,2);
%[albedo_2, normals_2] = estimate_alb_nrm(image_stack_2, scriptV);

%[image_stack_3, scriptV] = load_syn_images(image_dir,3);
%[albedo_3, normals_3] = estimate_alb_nrm(image_stack_3, scriptV);

%albedo(isnan(albedo)) = 0;
%normals(isnan(normals)) = 0;
%albedo_2(isnan(albedo_2)) = 0;
%normals_2(isnan(normals_2)) = 0;
%albedo_3(isnan(albedo_3)) = 0;
%normals_3(isnan(normals_3)) = 0;

%albedo = (albedo + albedo_2 + albedo_3)/3;
%albedo = cat(3,albedo,albedo_2,albedo_3);
%normals = cat(3,normals,normals_2,normals_3);
%normals = (normals + normals_2 + normals_3)/3;
%% integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
disp('Integrability checking')
[p, q, SE] = check_integrability(normals);
%For three channels
%[p_2, q_2, SE_2] = check_integrability(normals_2);
%[p_3, q_3, SE_3] = check_integrability(normals_3);

threshold = 0.005;
SE(SE <= threshold) = NaN; % for good visualization
fprintf('Number of outliers: %d\n\n', sum(sum(SE > threshold)));

%% compute the surface height
height_map = construct_surface( p, q);
%For Three channels
%height_map_2 = construct_surface( p_2, q_2);
%height_map_3 = construct_surface( p_3, q_3);

%height_map = (height_map + height_map_2 + height_map_3)/3;
%normals = (normals + normals_2 + normals_3)/3;

%% Display
show_results(albedo, normals, SE);
show_model(albedo, height_map);

%% Face
[image_stack, scriptV] = load_face_images('./yaleB02/');
[h, w, n] = size(image_stack);
fprintf('Finish loading %d images.\n\n', n);
disp('Computing surface albedo and normal map...')
[albedo, normals] = estimate_alb_nrm(image_stack, scriptV);

%% integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
disp('Integrability checking')
[p, q, SE] = check_integrability(normals);

threshold = 0.005;
SE(SE <= threshold) = NaN; % for good visualization
fprintf('Number of outliers: %d\n\n', sum(sum(SE > threshold)));

%% compute the surface height
height_map = construct_surface( p, q );

show_results(albedo, normals, SE);
show_model(albedo, height_map);

