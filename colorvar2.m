clear all;
clc;
close all;

load ('colorvar.mat', 'X_norm');
load ('th.mat', 'theta');
load ('colorvar.mat', 'y');
load ('colorvar.mat', 'Theta2');
load ('colorvar.mat', 'Theta1');


%% Use fminunc to minimize cost function by optimizing theta 

do 
  [J grad]=costfun(X_norm, theta, Theta1, Theta2, y); % J is cost; grad is gradient
  %options=optimset('GradObj', 'On');
  [theta fval info output]=fmincg(@(theta)costfun(X_norm, theta, Theta1, Theta2, y), theta);
  save ('th.mat', 'theta');
until (J < .00005);

