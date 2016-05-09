function [J grad]=costfun(theta, y, X_test, y_test, mini_batch_sz)

% computes the cost function and gradient for backpropagation

dbstop if error;
clc;




input_layer_size=size(X_test, 2);
hidden_layer_size=2000;  
hidden_layer_size2=2000;
num_colors=4;  % output layer size
m=size(X_test, 1);
Theta1=reshape(theta(1:hidden_layer_size*input_layer_size), hidden_layer_size, input_layer_size);
Theta2=reshape(theta(Theta1+1:Theta1+(hidden_layer_size2*(hidden_layer_size+1))), hidden_layer_size2, hidden_layer_size+1);
Theta3=reshape(theta(end-num_colors*(hidden_layer_size2+1)+1:end), num_colors, hidden_layer_size2+1);



% Feedforward and cost function

z=X_test*Theta1'; 
a_2=sigmoid(z); 
a_2=[ones(size(a_2, 1), 1) a_2];  %185x2001
z_2=a_2*Theta2'; 
a3=sigmoid(z_2);
a3=[ones(size(a3, 1), 1) a3]; %185x2001
z3=a3*Theta3';
h=sigmoid(z3); %185x3
y1=-y_test.*log(h); 
y1=sum(sum(y1));
y0=(1-y_test).*log(1-h); 
y0=sum(sum(y0));
J=(y1-y0)/mini_batch_sz; % Cost function for the training example selected


% Regularization of Cost Function to avoid overtraining

lambda=.3;
Theta1_reg=Theta1(:, 2:end);
Theta2_reg=Theta2(:, 2:end);
Theta3_reg=Theta3(:, 2:end);
theta_all=[Theta1_reg(:); Theta2_reg(:); Theta3_reg(:)].^2;
reg=sum(theta_all)*lambda/(2*mini_batch_sz);
J=J+reg; % Cost function with regularized weights


% Backpropagation

d1=zeros(size(Theta1));
d2=zeros(size(Theta2));
d3=zeros(size(Theta3));


if nargout > 1;
    err4=(h-y_test)'; %3x185
    err3=(Theta3(:, 2:end)'*err4).*sigmoidGradient(z_2)'; %2000x185
    err_2=(Theta2(:, 2:end)'*err3).*sigmoidGradient(z)'; %2000x185
    d3=d3+err4*a3; d3=d3(:, 2:end); %3x2000 
    d2=d2+err3*a_2; d2=d2(:, 2:end); %2000x2001
    d1=d1+err_2*X_test; d1=d1(:, 2:end);  
    Theta1_grad=1/m*d1+lambda/mini_batch_sz*Theta1(:, 2:end);
    Theta2_grad=1/m*d2+lambda/mini_batch_sz*Theta2(:, 2:end);
    Theta3_grad=1/m*d3+lambda/mini_batch_sz*Theta3(:, 2:end);  
    Theta1_grad=[zeros(size(Theta1_grad, 1), 1) Theta1_grad];
    Theta2_grad=[zeros(size(Theta2_grad, 1), 1) Theta2_grad];
    Theta3_grad=[zeros(size(Theta3_grad, 1), 1) Theta3_grad];
 
    %Regularized gradient
    grad=[Theta1_grad(:); Theta2_grad(:); Theta3_grad(:)];
end 



end 



