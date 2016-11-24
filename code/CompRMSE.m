function [ RMSE ] = CompRMSE( U, V, row, col, gndtruth )

predict = partXY(U', V, row, col, length(gndtruth));
RMSE = sqrt(sumsqr(predict' - gndtruth)/length(gndtruth));
end

