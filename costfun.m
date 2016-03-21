function [J grad]=costfun(X_norm, theta, Theta1, Theta2, y)

% computes the cost function and gradient for backpropagation


clc;

load colorvar.mat
%clear theta
%load th.mat

m=size(y, 1);  % number of training examples
hidden_layer_size=40;  
hidden_layer_size2=40;
num_colors=3;  % output layer size

%
%% feedforward
%X_norm(isnan(X_norm))=0;
%z2=X_norm*reshape(theta(1:hidden_layer_size*input_layer_size), hidden_layer_size, input_layer_size)';
%a2=sigmoid(z2);
%a2=[ones(size(a2, 1), 1) a2];
%z3=a2*reshape(theta(size(Theta1, 1)*size(Theta1, 2)+1:end), num_colors, hidden_layer_size+1)';
%h=sigmoid(z3);
%yvec=y(:);
%h=h';
%h=h(:);
%
%y1=-yvec'*log(h);
%y0=(1-yvec)'*log(1-h);
%J=(y1-y0)/m; % Cost function
%
%
%% Regularization of cost function
%lambda=.5;
%Theta1_reg=theta(hidden_layer_size+1:hidden_layer_size*input_layer_size);
%Theta2_reg=theta(hidden_layer_size*input_layer_size+4:end);
%theta_all=[Theta1_reg(:); Theta2_reg(:)];
%reg=sum(theta_all.^2)*lambda/(2*m);
%J=J+reg;
%
%d1=zeros(size(Theta1));
%d2=zeros(size(Theta2));
%if nargout>1;
%  for t=1:m
%		a_1=X_norm(t, :)';
%		z_2=reshape(theta(1:hidden_layer_size*input_layer_size), hidden_layer_size, input_layer_size)*a_1;
%		a_2=sigmoid(z_2);
%		a_2=[1; a_2];
%		z_3=reshape(theta(hidden_layer_size*input_layer_size+1:end), num_colors, hidden_layer_size+1)*a_2;
%		a_3=sigmoid(z_3);
%		err_out=zeros(m, 1);
%		err_out=a_3-y(:, t);
%		err_2=reshape(theta(hidden_layer_size*input_layer_size+1:end), num_colors, hidden_layer_size+1)'*err_out;
%		err_2=err_2(2:end).*sigmoidGradient(z_2);
%		d2=d2+err_out*a_2';
%		d1=d1+err_2*a_1';
%	endfor
%  Theta1_temp=[zeros(size(Theta1, 1), 1) reshape(theta(hidden_layer_size+1:hidden_layer_size*input_layer_size), hidden_layer_size, input_layer_size-1)];
%  Theta2_temp=[zeros(size(Theta2, 1), 1) reshape(theta(hidden_layer_size*input_layer_size+num_colors+1:end), num_colors, hidden_layer_size)];
%  Theta1_grad=1/m*d1+lambda/m*Theta1_temp;
%  Theta2_grad=1/m*d2+lambda/m*Theta2_temp;
%  grad=[Theta1_grad(:); Theta2_grad(:)]; 
%  grad(isnan(grad))=0;
%endif
mini_batch_sz=1;
% Randomly select 1 training example for stochastic gradient descent
allvals=[y' X_norm];
allvals=allvals(randperm(mini_batch_sz), :);
y_test=allvals(:, 1:3); % 1x3 matrix
X_test=allvals(:, 4:end); % 1x42571 matrix
X_test(isnan(X_test))=0;


% Feedforward and cost function
z=X_test*reshape(theta(1:numel(Theta1)), hidden_layer_size, input_layer_size)';
a_2=sigmoid(z); 
a_2=[ones(size(a_2, 1), 1) a_2]; %1x41 matrix
z_2=a_2*reshape(theta(numel(Theta1)+1:numel(Theta1)+numel(Theta2)), hidden_layer_size2, hidden_layer_size+1)';
a3=sigmoid(z_2);
a3=[ones(size(a3, 1), 1) a3]; %1x41 matrix
z3=a3*reshape(theta((end-numel(Theta3))+1:end), num_colors, hidden_layer_size2+1)';
h=sigmoid(z3); %1x3 matrix
y1=-y_test*log(h)'; 
y0=(1-y_test)*log(1-h)';
J=(y1-y0)/mini_batch_sz; % Cost function for the training example selected

% Regularization of Cost Function
lambda=.2;
Theta1_reg=theta(hidden_layer_size+1:hidden_layer_size*input_layer_size);
Theta2_reg=theta(hidden_layer_size*input_layer_size+hidden_layer_size2+1:hidden_layer_size*hidden_layer_size2);
Theta3_reg=theta(end-(num_colors*hidden_layer_size2-4):num_colors+1);
theta_all=[Theta1_reg(:); Theta2_reg(:); Theta3_reg(:)];
reg=sum(theta_all.^2)*lambda/(2*mini_batch_sz);
J=J+reg; % Cost function with regularized weights

% Backpropagation
%for t=1:mini_batch_sz
d1=zeros(size(Theta1));
d2=zeros(size(Theta2));
d3=zeros(size(Theta3));


if nargout > 1;
  err_out=h-y_test; %1x3 matrix
  err_2=err_out*reshape(theta(end-numel(Theta3)+1:end), num_colors, hidden_layer_size2+1);
  err_2=err_2(2:end).*sigmoidGradient(z_2);
  err_3=err_2*reshape(theta(numel(Theta1)+1:numel(Theta1)+numel(Theta2)), hidden_layer_size2, hidden_layer_size+1);
  err_3=err_3(2:end).*sigmoidGradient(z);
  d3=d3+err_out'*a3;
  d2=d2+err_2'*a_2;
  d1=d1+err_3'*X_test;
  Theta1_temp=[zeros(size(Theta1, 1), 1) reshape(theta(hidden_layer_size+1:numel(Theta1)), hidden_layer_size, input_layer_size-1)];
  Theta2_temp=[zeros(size(Theta2, 1), 1) reshape(theta(numel(Theta1)+size(Theta2, 1)+1:numel(Theta1)+numel(Theta2)), hidden_layer_size2, hidden_layer_size)];
  Theta3_temp=[zeros(size(Theta3, 1), 1) reshape(theta(end-numel(Theta3)+1+num_colors:end), num_colors, hidden_layer_size2)];
  Theta1_grad=1/mini_batch_sz*d1+lambda/mini_batch_sz*Theta1_temp;
  Theta2_grad=1/mini_batch_sz*d2+lambda/mini_batch_sz*Theta2_temp;
  Theta3_grad=1/mini_batch_sz*d3+lambda/mini_batch_sz*Theta3_temp;
  grad=[Theta1_grad(:); Theta2_grad(:); Theta3_grad(:)];  % gradient for each theta
  grad(isnan(grad))=0;
  
%
%    a_1=X_test'; % 42571x1 vector
%    z_2=reshape(theta(1:hidden_layer_size*input_layer_size), hidden_layer_size, input_layer_size)*a_1; %50x42571 matrix x 42571x1 matrix
%    a_2=sigmoid(z_2); % 50x1 matrix
%    a_2=[1; a_2];
%    z_3=reshape(theta(hidden_layer_size*input_layer_size+1:end), num_colors, hidden_layer_size+1)*a_2; %3x51 matrix x 51x1 matrix
%    a_3=sigmoid(z_3); % 3x1 matrix
%    err_out=a_3-y_test; % 3x1 matrix
%    err_2=reshape(theta(hidden_layer_size*input_layer_size+1:end), num_colors, hidden_layer_size+1)'*err_out; % 51x3 matrix x 3x1 matrix
%    err_2=err_2(2:end).*sigmoidGradient(z_2);
%    d2=d2+err_out*a_2';
%    d1=d1+err_2*a_1';
%    Theta1_temp=[zeros(size(Theta1, 1), 1) reshape(theta(hidden_layer_size+1:hidden_layer_size*input_layer_size), hidden_layer_size, input_layer_size-1)];
%    Theta2_temp=[zeros(size(Theta2, 1), 1) reshape(theta(hidden_layer_size*input_layer_size+num_colors+1:end), num_colors, hidden_layer_size)];
%    Theta1_grad=1/mini_batch_sz*d1+lambda/m*Theta1_temp;
%    Theta2_grad=1/mini_batch_sz*d2+lambda/m*Theta2_temp;
%    grad=[Theta1_grad(:); Theta2_grad(:)]; 
%    grad(isnan(grad))=0;
endif 
%theta=theta-.001*grad*J; 
%endfor;
end 



