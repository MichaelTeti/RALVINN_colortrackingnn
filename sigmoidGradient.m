function g = sigmoidGradient(z)


g = zeros(size(z));


g=1./(1+e.^-z);
g=g.*(1-g);





end
