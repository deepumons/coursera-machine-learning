function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

least_error = 1;
C_least = 0;
sigma_least = 0;

for c_test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

	for sigma_test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
	
	model = svmTrain(X, y, c_test, @(X, y) gaussianKernel(X, y, sigma_test));
	
	predictions = svmPredict(model, Xval);
	
	error = mean(double(predictions ~= yval))
	
	%fprintf("Error:%f\tC:%f\tsigma:%f\t", error, c_test, sigma_test);
	
		if (error < least_error)
			least_error = error;
			C_least = c_test;
			sigma_least = sigma_test;
		end

	end
	
end


C = C_least;
sigma = sigma_least;



% =========================================================================

end
