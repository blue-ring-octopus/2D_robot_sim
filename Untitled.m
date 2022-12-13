clear all;
close all;

x_d=zeros(3,5);
x_act=zeros(3,5);
u_act=[7.24, 7.65, 6.36, 7.34;
    -1.47, -1.98, -1.71, -1.46];
u_d=[5*sqrt(2);
    -pi/2];


for i=1:4
    x_act(:,i+1)= x_act(:,i)+[u_act(1,i)*cos(x_act(3,i)+1/2*u_act(2,i))
                      u_act(1,i)*sin(x_act(3,i)+1/2*u_act(2,i))
                      u_act(2,i)];
  x_act(3,i+1)=atan2(sin(x_act(3,i+1)), cos(x_act(3,i+1)));
  
  x_d(:,i+1)= x_d(:,i)+[u_d(1)*cos(x_d(3,i)+1/2*u_d(2))
                      u_d(1)*sin(x_d(3,i)+1/2*u_d(2))
                      u_d(2)];
  x_d(3,i+1)=atan2(sin(x_d(3,i+1)), cos(x_d(3,i+1)));
end

figure(1)
plot(x_act(1,:),x_act(2,:), "--", "lineWidth",2)
hold on
plot(x_d(1,:),x_d(2,:),  "lineWidth",2)
legend('actual ',' expected')

landmark=[5, 5, -5,-5, 5;
    0,-10, -10,0, 0];
Q=[0.01,0;0,0.01];

% z=zeros(2,5);
% for i=1:5
%  mu=[norm(x_d(1:2,i)-landmark(:,i) ); 
%      atan2(landmark(2,i)-x_d(2,i), landmark(1,i)-x_d(1,i))-x_act(3,i)]
%  z(:,i)=mvnrnd(mu,Q)
% end

z=[5.03,5.07,4.42,4.39,1.57;
    0.03,-0.17,0.45,1.01,1.00];

H=[cos(0.03) -5.03*sin(0.03);
    sin(0.03) 5.03*cos(0.03)];
mu_m=[5.03*cos(0.03);
    5.03*sin(0.03)];
figure(2)
plot(5,-5, "o", "lineWidth",2)
hold on
plot(mu_m(1),mu_m(2), "x", "lineWidth",2)

plotErrorEllipse(mu_m, H*Q*H', 0.68)
plotErrorEllipse([5,-5], [0.675 -0.575;-0.575  0.675], 0.68)

title("t=1")
axis([-8 8 -12 4])

%%
close all
dat=load('example.mat')
figure(3)
for i=1:5
mu=cell2mat(dat.mu(i))
sigma=cell2mat(dat.sigma(i))
plot(mu(1),mu(2), "bo", "lineWidth",2)
hold on
plotErrorEllipse(mu, sigma(1:2, 1:2) , 0.68)
plot(landmark(1,:), landmark(2,:), "kx", "lineWidth",2,"MarkerSize",10)
plot(x_act(1,:),x_act(2,:), "--", "lineWidth",2)
feature=mu(4:length(mu));
feature_sigma=sigma(4:length(mu),4:length(mu));
for j=1:length(feature)/2
plot(feature(2*j-1), feature(2*j), 'bx', "lineWidth",2,"MarkerSize",10)
plotErrorEllipse([feature(2*j-1), feature(2*j)], feature_sigma(2*j-1:2*j, 2*j-1:2*j) , 0.68)
end
axis([-8 8 -12 4])
hold off
pause
end

function plotErrorEllipse(mu, Sigma, p)

    s = -2 * log(1 - p);

    [V, D] = eig(Sigma * s);

    t = linspace(0, 2 * pi);
    a = (V * sqrt(D)) * [cos(t(:))'; sin(t(:))'];

    plot(a(1, :) + mu(1), a(2, :) + mu(2),"lineWidth",2 );
end