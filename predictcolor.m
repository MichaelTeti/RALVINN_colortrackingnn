function [p wrong]=predictcolor(y_predict, allvalstest, theta)
% Predicts the color present in the picture it is shown

inputsz=size(allvalstest, 2);
hidden1sz=2000;
hidden2sz=2000;
outputsz=4;
theta11=inputsz*hidden1sz;
theta22=(hidden1sz+1)*hidden2sz;
theta33=outputsz*(hidden2sz+1);

a2=sigmoid(allvalstest*reshape(theta(1:theta11), hidden1sz, inputsz)');
a2=[ones(size(a2, 1), 1) a2];
a3=sigmoid(a2*reshape(theta(theta11+1:theta11+theta22), hidden2sz, hidden1sz+1)');
a3=[ones(size(a3, 1), 1) a3];
output=sigmoid(a3*reshape(theta(end-theta33+1:end), outputsz, hidden2sz+1)');
[d f]=max(output, [], 2);
[h g]=max(y_predict, [], 2);
e=0;
wrong=[];
for i=1:size(output, 1);
    if f(i)==g(i)
        e=e+1;
    elseif f(i)~=g(i)
        wrong(i, 1)=f(i);
        wrong(i, 2)=g(i);
    end
end

p=e/size(output, 1)*100;

    
    



    
    
    
    
    
    
