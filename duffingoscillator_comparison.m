%% compare different kernels for sobolev space by accuracy of their predictions for Duffing oscillator
%% note sobolev spaces kernels all equivalent norms w not necessarily known constants so cannot compare residuals directly

clear

addpath(genpath('./algorithms'))

rng(0) %rng(0) nice, rng(1),2 not

no_kernels=6;

%% generate snapshot data
time_step=0.01;
steps=30;
no_traj=40;
M=no_traj*steps;
x0=2*rand(2,no_traj+10)-1;
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

%% generate test trajectories for display
use=6; %1 and 6
[~, x_test]=ode45(@(t,x) duffing(t,x),0:time_step:(steps*time_step),x0(:,no_traj+use));
x_test=x_test';

%% initialise variables
x_start=x0(:,no_traj+use);
G_start=zeros(1,M);
x_kedmd=zeros(2,steps+1,no_kernels);

%% compute forecast errors for various different kernels
for k=1:no_kernels
    %compute operator folding matrices
    ker=@(x,t) kernel_comp(x,t,k);
    [G,A,R]=generate_matrices(x,y,ker);
    
    %compute evals
    [~,Lambda,W]=eig(A,G,'vector');

    %get initial values
    for i=1:M
        G_start(i)=ker(x_start,x(:,i));
    end

    % compute KMD and use to predict future behaviour
    mode_full=((G*W)\(x.')).';
    psi0_full=G_start*W;
    x_kedmd(:,:,k)=real(transpose(transpose(psi0_full).*(conj(Lambda).^(0:steps)))*mode_full.').';
end

%% Plot errors for predictions given by the different kernels
colours=turbo(no_kernels);
data=x_test(:,1:steps+1);
er=zeros(no_kernels);
figure
for i=1:no_kernels
    er = sum(abs(x_kedmd(:,:,i)-data).^2)./sum(abs(data).^2,1);
    semilogy(0:0.01:(0.01*steps),er,'linewidth',2)
    hold on
end
grid on
axis tight

title('Relative forecast error comparison','fontsize',18,'interpreter','latex')
xlabel('Time (s)','interpreter','latex','fontsize',18)
ylabel('Relative Forecast Error','interpreter','latex','fontsize',18)
legend({'Wendland, $k=1$','Wendland, $k=2$','Wendland, $k=3$','Mat\''ern, $n=2$','Mat\''ern, $n=3$','Mat\''ern, $n=4$'},'interpreter','latex','fontsize',16,'location','northeast')
exportgraphics(gcf,'duffing_forecast_error.pdf','ContentType','vector','BackgroundColor','none')


%% define kernels
function ker=kernel_comp(x,t,num)
    r=vecnorm(x-t);
    if num==1
        %wendland equiv to H^(5/2)
        r=r*0.0098; %parameters based on result of duffingoscillator_hyperparameter
        l=3;
        r=min(r,1);
        ker=(1-r)^(l+1)*((l+1)*r+1);
    elseif num==2
        %wendland equiv to H^(7/2)
        r=r*(0.256);
        l=4;
        r=min(r,1);
        ker=(1-r)^(l+2)*((l^2+4*l+3)*r^2+(3*l+6)*r+3);
    elseif num==3
        %wendland equiv to H^(9/2)
        l=5;
        r=r*1.06;
        r=min(r,1);
        ker=(1-r)^(l+3)*((l^3+9*l^2+23*l+15)*r^3+(6*l^2+36*l+45)*r^2+(15*l+45)*r+15);
    elseif num==4
        %matern equiv to H^2
        r=r*(0.0003);
        ker=zeros(1,size(x,2));
        ker(r>0)=(sigma*r(r>0)).*besselk(-1,sigma*r(r>0));
        ker(r==0)=1;
    elseif num==5
        %matern equiv to H^3
        r=r*(0.3);
        ker=zeros(1,size(x,2));
        ker(r>0)=(sigma*r(r>0)).^(2).*besselk(-2,sigma*r(r>0));
        ker(r==0)=2;
    elseif num==6
        %matern equiv to H^3
        r=r*(4.1);
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