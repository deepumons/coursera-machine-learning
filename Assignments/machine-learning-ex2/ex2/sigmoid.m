function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
%g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

%fprintf('z size: %i', size(z))
%fprintf('\n')

g =  ones(size(z)) ./ (1 + e.^ -z);


% =============================================================

end
