clear all;
clc;


load colorvar.mat




%% Use fminunc to minimize cost function by optimizing theta 

do 
  [J grad]=costfun(X_norm, theta, Theta1, Theta2, y); % J is cost; grad is gradient
  options=optimset('GradObj', 'On');
  [v fval info output]=fminunc(@(theta)costfun(X_norm, theta, Theta1, Theta2, y), theta, options);
  theta=v;
  save('th.mat', 'theta');
until (J < .00005);

