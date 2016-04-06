% Color identification RALVINN
% Author = Michael Teti, MPCR Lab, FAU Center for Complex Systems
% 2/28/16

clear all;
clc;
dbstop if error;


% Go to directory where green pictures and files are located
cd ('C:/Users/Michael/Python1/MT_pics/Green');

% choose all files in green folder with .png extension, convert to gray, and crop
green_contents=dir('*.png');
all_colors=ones(1, 42570);
for i=1:numel(green_contents);
	filename=green_contents(i).name;
	[path name]=fileparts(filename);
	im=imread(filename);
	imgray=rgb2gray(im);
	imcrop=imgray(150:278, 1:330);
	greenvec=imcrop(:)';
	all_colors=[all_colors; greenvec];
end;

% store cropped images as a row in matrix all_colors
all_colors=all_colors(2:end, :);

% Perform same steps above for pink
cd ('C:/Users/Michael/Python1/MT_pics/Pink');
pink_contents=dir('*.png');

for i=1:numel(pink_contents);
	filename=pink_contents(i).name;
	[path name]=fileparts(filename);
	im=imread(filename);
	imgray=rgb2gray(im);
	imcrop=imgray(150:278, 1:330);
	pinkvec=imcrop(:)';
	all_colors=[all_colors; pinkvec];
end;

% Perform same steps for yellow folder
cd ('C:/Users/Michael/Python1/MT_pics/Yellow');
yellow_contents=dir('*.png');

for i=1:numel(yellow_contents);
	filename=yellow_contents(i).name;
	[path name]=fileparts(filename);
	im=imread(filename);
	imgray=rgb2gray(im);
	imcrop=im(150:278, 1:330);
	yellowvec=imcrop(:)';
	all_colors=[all_colors; yellowvec];
end;


% Feature Scaling



mu=zeros(1, size(all_colors, 2));
sigma=zeros(1, size(all_colors, 2));
all_colors=im2double(all_colors);
[a, b]=find(all_colors==0);
for i=1:length(a);
  all_colors(a(i), b(i))=10^-10;
endfor;

for j=1:size(all_colors, 2);
	mu(1, j)=mean(all_colors(:, j));
	sigma(1, j)=std(all_colors(:, j));
	all_colors(:, j)=(all_colors(:, j)-mu(1, j))./sigma(1, j);
end;


% Initialize parameters and other values

X_norm=all_colors;
X_norm=[ones(size(X_norm, 1), 1) X_norm];
num_colors=3;
input_layer_size=size(X_norm, 2);
hidden_layer_size=50;
y=zeros(num_colors, size(green_contents, 1)+size(pink_contents, 1)+size(yellow_contents, 1));
y(1, 1:size(green_contents, 1))=1; % Node 1 "fires" for green colors
y(2, size(green_contents, 1)+1:size(green_contents, 1)+size(pink_contents, 1))=1; % second output node is pink
y(3, end-size(yellow_contents, 1):end)=1; % 3rd output node fires for yellow
m=size(y, 2);

% Initialize weights between nodes
epsilon1=sqrt(6)/sqrt(input_layer_size+hidden_layer_size);
epsilon2=sqrt(6)/sqrt(hidden_layer_size+1+num_colors);
Theta1=rand(hidden_layer_size, input_layer_size)*2*epsilon1-epsilon1;
Theta2=rand(num_colors, hidden_layer_size+1)*2*epsilon2-epsilon2;
theta=[Theta1(:); Theta2(:)];


cd ('C:/Users/Michael/Python1/MT_pics');
[J grad]=costfun(X_norm, theta, Theta1, Theta2, hidden_layer_size, input_layer_size, num_colors, m, y);




