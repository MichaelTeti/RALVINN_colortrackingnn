% Color Tracking RALVINN 
% Version 2
% Michael A. Teti
% 5/3/16
% FAU MPCR Lab

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all
clc
dbstop if error

%Load Data

cd ('/home/mpcr/Desktop/MT_pics/pics');

downsample=3;
all_colors=[];
basepath=pwd; % assign working directory to basepath
all_paths=dir(basepath);  % assign all subfolders in working dir. to all_paths
subfolds=[all_paths(:).isdir]; % get only subfolders
foldersNames = {all_paths(subfolds).name}';
foldersNames(ismember(foldersNames,{'.','..'})) = []; 
for i=1:length(foldersNames), %loop through all folders
    tmp=foldersNames{i};  %get folder by index
    p=strcat([basepath '/']); 
    currentPath=strcat([p tmp]); % add base to current folder
    cd(currentPath);   % change directory to new path
    files=dir('*.png'); % list all images in your path 
    for j=1:length(files), % loop through images 
        im=imread(files(j).name); % read each image 
        im=im2double(im);
	    im=im(150:278, 1:330, :);
	    im=im(1:downsample:end, 1:downsample:end, :);  %downsample to speed things up
	    im2=im+.03*rand(size(im)); %add noise to create more training images
	    im3=im; %add brightness to increase range a little more
        im3(1:20:end, 1:20:end, :)=0.0010;
	    all_colors=[all_colors im(:) im2(:) im3(:)];
    end
end

all_colors=all_colors';
cd ('/home/mpcr/Desktop/MT_pics');

[a b]=find(all_colors==0);
all_colors(a, b)=0.0001;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Feature Scaling 

mu=mean(all_colors);
sigma=std(all_colors);
for i=1:size(all_colors, 2);
	all_colors(:, i)=(all_colors(:, i)-mu(i))./sigma(i);
end


%Initialize Parameters and other values

blue_contents=45; %45 blue images in folder
green_contents=55; %55 green images in folder
pink_contents=55;  %55 pink images in folder
yellow_contents=48; %48 yellow images in folder
X_norm=all_colors;
X_norm=[ones(size(X_norm, 1), 1) X_norm];  %add bias node to input
num_colors=4;  %number of output nodes 
input_layer_size=size(X_norm, 2);  %number of input nodes/input features
hidden_layer_size=2000;
hidden_layer_size2=2000;
y=zeros(num_colors, (green_contents+pink_contents+yellow_contents+blue_contents)*3);
y(1, 1:blue_contents*3)=1; % Node 1 "fires" for blue colors
y(2, blue_contents*3+1:blue_contents*3+1+green_contents*3)=1; % second output node is green
y(3, blue_contents*3+1+green_contents*3+1:blue_contents*3+green_contents*3+pink_contents*3)=1; % 3rd output node fires for pink
y(4, (end-yellow_contents*3)+1:end)=1; %4th node outputs yellow
m=size(y, 2); %number of training examples


% Initialize theta
epsilon1=sqrt(6)/sqrt(input_layer_size+hidden_layer_size);
epsilon2=sqrt(6)/sqrt(hidden_layer_size+1+hidden_layer_size2);
epsilon3=sqrt(6)/sqrt(hidden_layer_size2+1+num_colors);
Theta1=rand(hidden_layer_size, input_layer_size)*2*epsilon1-epsilon1;
Theta2=rand(hidden_layer_size2, hidden_layer_size+1)*2*epsilon2-epsilon2;
Theta3=rand(num_colors, hidden_layer_size2+1)*2*epsilon3-epsilon3;
theta=[Theta1(:); Theta2(:); Theta3(:)];


% Randomly select half of training images for gradient descent

mini_batch_sz=ceil(size(X_norm, 1)/2);
allvals=[y' X_norm];
allvals=allvals(randperm(size(all_colors, 1)), :);
allvalstraining=allvals(1:mini_batch_sz, num_colors+1:end);
allvalstest=allvals(mini_batch_sz+1:end, num_colors+1:end);
y_test=allvals(1:mini_batch_sz, 1:num_colors); % 62x3 matrix
X_test=allvalstraining; % 62x2740 
y_predict=allvals(mini_batch_sz+1:end, 1:num_colors);
X_test(isnan(X_test))=0.0001;


% Compute Cost function and backpropagation

[J grad]=costfun(theta, y, X_test, y_test, mini_batch_sz); % J is cost; grad is gradient
[theta fval info]=fmincg(@(theta)costfun(theta, y, X_test, y_test, mini_batch_sz), theta); 


% Test untrained portion of data and return percent correct
[p wrong]=predictcolor(y_predict, allvalstest, theta)












