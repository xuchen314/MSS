function [ RMSE ] = CompRMSEm( X, loc, gndtruth )  
  predict = X(loc);
  RMSE = sqrt(sumsqr(predict - gndtruth)/length(gndtruth));
end

