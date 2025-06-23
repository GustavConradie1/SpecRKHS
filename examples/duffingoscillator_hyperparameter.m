%% hyperparameter grid search for scale parameters for duffing oscillator kernels

addpath(genpath('./algorithms'))

rng(5)

%% number of kernel (wrt kernel_comp definition to be considered)
no=6;
scale_range=[10^(-4) 10^(-2) 1 10^2 10^4];
no_test=10;

%% generate snapshot data
time_step=0.01;
steps=30;
no_traj=40;
M=no_traj*steps;
x0=2*rand(2,no_traj+no_test)-1;
x=zeros(2,M);
y=zeros(2,M);
for i=1:no_traj
    [~, x1]=ode45(@(t,x) duffing(t,x),0:time_step:(steps*time_step),x0(:,i));
    x1=x1';
    for j=1:steps
        x(:,(i-1)*steps+j)=x1(:,j);
        y(:,(i-1)*steps+j)=x1(:,j+1);
    end
end

%% generate test trajectory
x_test=zeros(2,steps+1,no_test);
for j=1:no_test
    [~, x1]=ode45(@(t,x) duffing(t,x),0:time_step:(steps*time_step),x0(:,no_traj+j));
    x_test(:,:,j)=x1';
end

%% initialise variables
error=zeros(no_test,length(scale_range));

%% compute forecast errors for various different kernels
for k=1:length(scale_range)
    %compute operator folding matrices
    ker=@(x,t) kernel_comp(x,t,scale_range(k),no);
    [G,A,R]=generate_matrices(x,y,ker);
    cond(G)
    
    %compute all evals
    [~,Lambda,W]=eig(A,G,'vector');

    for j=1:no_test
        x_start=x0(:,no_traj+j);
        G_start=zeros(1,M);

        %get initial values
        for i=1:M
            G_start(i)=ker(x_start,x(:,i));
        end

        % compute kedmd prediction of trajectory
        mode_full=((G*W)\(x.')).';
        psi0_full=G_start*W;
        x_kedmd=real(transpose(transpose(psi0_full).*(conj(Lambda).^(0:steps)))*mode_full.').';

        %compute mean error across trajectories
        data=x_test(:,:,j);
        error(j,k) = mean(sum(abs(x_kedmd-data).^2)./sum(abs(data).^2,1));
    end
end

%% output results
for k=1:length(scale_range)
    fprintf('Scale was %f, mean error was %g \n',scale_range(k),mean(error(:,k)))
end

%% define kernels
function ker=kernel_comp(x,t,scale,num)
    r=scale*vecnorm(x-t);
    if num==1
        %wendland equiv to H^(5/2)
        l=3;
        r=min(r,1);
        ker=(1-r)^(l+1)*((l+1)*r+1);
    elseif num==2
        %wendland equiv to H^(7/2)
        l=4;
        r=min(r,1);
        ker=(1-r)^(l+2)*((l^2+4*l+3)*r^2+(3*l+6)*r+3);
    elseif num==3
        %wendland equiv to H^(9/2)
        l=5;
        r=min(r,1);
        ker=(1-r)^(l+3)*((l^3+9*l^2+23*l+15)*r^3+(6*l^2+36*l+45)*r^2+(15*l+45)*r+15);
    elseif num==4
        %matern equiv to H^2
        ker=zeros(1,size(x,2));
        ker(r>0)=(sigma*r(r>0)).*besselk(-1,sigma*r(r>0));
        ker(r==0)=1;
    elseif num==5
        %matern equiv to H^3
        ker=zeros(1,size(x,2));
        ker(r>0)=(sigma*r(r>0)).^(2).*besselk(-2,sigma*r(r>0));
        ker(r==0)=2;
    elseif num==6
        %matern equiv to H^4
        ker=zeros(1,size(x,2));
        ker(r>0)=(sigma*r(r>0)).^(3).*besselk(-3,sigma*r(r>0));
        ker(r==0)=8;
    end
end

%% define duffing oscillator
function dxdt = duffing(~,x)
    dxdt=zeros(2,1);
    dxdt(1)=x(2);
    dxdt(2)=-0.2*x(2)-x(1)-x(1).^3;
end