clc;
clear;
close;
rng(10);

% Initialize data
[X, Y] = generateData();

% PLA
w = PLA_train(X, Y);

% Draw points
hold on;
scatter(X(1, Y > 0), X(2, Y > 0), 25, 'b', 'filled');
scatter(X(1, Y < 0), X(2, Y < 0), 25, 'r', 'filled');

% Draw plane
fimplicit(@(x, y) w(1) * x + w(2) * y + w(3), 'k-', 'LineWidth', 1.2);

% Others
axis([-0.1, 1.1, -0.1, 1.1]);
xx = 0.5;
yy = (-w(3) - w(1)*xx) / w(2) + 0.03;

legend('$y_i=1$', '$y_i=-1$', 'location', 'best', 'Interpreter', 'latex');
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
text(xx, yy, '$\mathbf w^T\mathbf x+b = 0$', 'Interpreter', 'latex');

function w = PLA_train(X, Y)
MAX_ITER = 90;
eta = 0.1;

% data preprocess
[d, n] = size(X);
X1 = [X; ones(1, n)];
XY = X1.*Y';

% compute w
w = randn(d+1, 1);
for i = 1:MAX_ITER
    mask = find(w'*XY < 0);
    if isempty(mask)
        break;
    end
    w = w + eta * sum(XY(:, mask), 2);
    
    % Process
    hold on;
    axis([-0.1, 1.1, -0.1, 1.1]);
    scatter(X(1, Y > 0), X(2, Y > 0), 25, 'b', 'filled');
    scatter(X(1, Y < 0), X(2, Y < 0), 25, 'r', 'filled');
    fimplicit(@(x, y) w(1) * x + w(2) * y + w(3), 'k-', 'LineWidth', 1.2);
    drawnow;
    cla;
end

end


function [X, Y] = generateData()
data = rand(2, 25);

maskPos = data(1, :) + data(2, :) > 1.2;
maskNeg = data(1, :) + data(2, :) < 0.8;

Xpos = data(:, maskPos);
Xneg = data(:, maskNeg);
X = [Xpos, Xneg];
Y = [ones(size(Xpos, 2), 1); -ones(size(Xneg, 2), 1)];

end

function [X, Y] = generateDataNoise()
rng(10);
data = rand(2, 25);

maskPos = data(1, :) + data(2, :) > 1.1;
maskNeg = data(1, :) + data(2, :) < 0.9;

Xpos = data(:, maskPos);
Xneg = data(:, maskNeg);
X = [Xpos, Xneg];
temp = X(:, 1);
X(:, 1) = X(:, 15);
X(:, 15) = temp;
Y = [ones(size(Xpos, 2), 1); -ones(size(Xneg, 2), 1)];

end