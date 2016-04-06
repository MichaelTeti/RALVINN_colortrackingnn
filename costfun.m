function [J grad]=costfun(X_norm, theta, Theta1, Theta2, y)

% computes the cost function and gradient for backpropagation
%debug_on_warning(1);
debug_on_error(1);
clear all;
clc;

load ('colorvar.mat', 'X_norm'); %124*42571
load ('colorvar.mat', 'y'); %3x124 
load ('colorvar.mat', 'Theta1');
load ('colorvar.mat', 'Theta2');
load ('colorvar.mat', 'Theta3');
load ('th.mat', 'theta');



input_layer_size=size(X_norm, 2);
hidden_layer_size=1000;  
hidden_layer_size2=1000;
num_colors=3;  % output layer size


mini_batch_sz=1;
m=size(X_norm, 1);

% Randomly select 50 training example for stochastic gradient descent
allvals=[y' X_norm];
allvals=allvals(randperm(mini_batch_sz), :);
y_test=allvals(:, 1:3); % 50x3 matrix
X_test=allvals(:, 4:end); % 50x2740 matrix
X_test(isnan(X_test))=0;



% Feedforward and cost function
z=X_test*reshape(theta(1:numel(Theta1)), hidden_layer_size, input_layer_size)';
a_2=sigmoid(z); 
a_2=[ones(size(a_2, 1), 1) a_2]; %50x1001 matrix
z_2=a_2*reshape(theta(numel(Theta1)+1:numel(Theta1)+numel(Theta2)), hidden_layer_size2, hidden_layer_size+1)';
a3=sigmoid(z_2);
a3=[ones(size(a3, 1), 1) a3]; %50x1001 matrix
z3=a3*reshape(theta((end-numel(Theta3))+1:end), num_colors, hidden_layer_size2+1)';
h=sigmoid(z3); %50x3 matrix
y1=-y_test.*log(h); %50x3 matrix
y1=sum(sum(y1));
y0=(1-y_test).*log(1-h); %50x3 matrix
y0=sum(sum(y0));
J=(y1-y0)/mini_batch_sz; % Cost function for the training example selected

% Regularization of Cost Function
lambda=.003;
Theta1_reg=theta(hidden_layer_size+1:hidden_layer_size*input_layer_size).^2;
Theta2_reg=theta(hidden_layer_size*input_layer_size+hidden_layer_size2+1:hidden_layer_size*hidden_layer_size2).^2;
Theta3_reg=theta(end-(num_colors*hidden_layer_size2-4):num_colors+1).^2;
theta_all=[Theta1_reg(:); Theta2_reg(:); Theta3_reg(:)];
reg=sum(theta_all)*lambda/(2*mini_batch_sz);
J=J+reg; % Cost function with regularized weights

% Backpropagation
%for t=1:mini_batch_sz
d1=zeros(size(Theta1));
d2=zeros(size(Theta2));
d3=zeros(size(Theta3));


if nargout > 1;
  for t=1:mini_batch_sz
    err4=h(t, :)'-y_test(t, :)'; %3x1 matrix
    err3=reshape(theta(end-numel(Theta3)+1:end), num_colors, hidden_layer_size2+1)'*err4; %1001x1 matrix
    err3=err3(2:end).*sigmoidGradient(z_2(t, :))'; %1000x1 matrix
    err_2=reshape(theta(numel(Theta1)+1:numel(Theta1)+numel(Theta2)), hidden_layer_size2, hidden_layer_size+1)'*err3; %1000x1 matrix
    %err_2=err3*reshape(theta(numel(Theta1)+1:numel(Theta1)+numel(Theta2)), hidden_layer_size2, hidden_layer_size+1);
    err_2=err_2(2:end).*sigmoidGradient(z(t, :))'; 
    d3=d3+err4*a3(t, :);
    d2=d2+err3*a_2(t, :);
    d1=d1+err_2*X_test(t, :);
  endfor;
    Theta1_temp=[zeros(size(Theta1, 1), 1) reshape(theta(hidden_layer_size+1:numel(Theta1)), hidden_layer_size, input_layer_size-1)];
    Theta2_temp=[zeros(size(Theta2, 1), 1) reshape(theta(numel(Theta1)+size(Theta2, 1)+1:numel(Theta1)+numel(Theta2)), hidden_layer_size2, hidden_layer_size)];
    Theta3_temp=[zeros(size(Theta3, 1), 1) reshape(theta(end-numel(Theta3)+1+num_colors:end), num_colors, hidden_layer_size2)];
    Theta1_grad=1/m*d1+lambda/mini_batch_sz*Theta1_temp;
    Theta2_grad=1/m*d2+lambda/mini_batch_sz*Theta2_temp;
    Theta3_grad=1/m*d3+lambda/mini_batch_sz*Theta3_temp;
    grad=[Theta1_grad(:); Theta2_grad(:); Theta3_grad(:)];  % gradient for each theta
endif 
%theta=theta-.001*grad*J; 
end 



