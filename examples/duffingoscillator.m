%% for the optimal kernel found using duffingoscillator_hyperparameter, compute verified eigenpairs and pseudospectra

addpath(genpath('./algorithms'))
addpath(genpath('./colormaps'))

rng(3)

%% generate snapshot data
time_step=0.01;
steps=30;
no_traj=40;
M=no_traj*steps;
x0=2*rand(2,no_traj)-1;
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

%% compute operator folding matrices
ker=@(x,t) kernel(x,t);
[G,A,R]=generate_matrices(x,y,ker);

%% compute verified eigenvalues
[~,~,Lambda,~,res,~]=verified_eigenvalues(G,A,R,0.1);

%% plot spurious and verified eigenvalues
figure
scatter(real(Lambda),imag(Lambda),200,res,'.','LineWidth',1);
box on
hold on
clim([0,1])
load('cmap.mat')
colormap(cmap2); colorbar
xlabel('$\mathrm{Re}(\lambda)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{Im}(\lambda)$','interpreter','latex','fontsize',18)
title('Residuals of Duffing oscillator','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18; axis equal; axis([-1.2 1.2 -1.2 1.2])
plot(sin(0:0.01:2*pi), cos(0:0.01:2*pi),'-k','LineWidth',1);
exportgraphics(gcf,'duffing_oscillator_evals.pdf','ContentType','vector','BackgroundColor','none')

%% compute psuedospectra
pts=50;
x_pts=linspace(-1.5,1.5,pts);    y_pts=linspace(0,1.5,pts/2);
z_pts=kron(x_pts,ones(length(y_pts),1))+1i*kron(ones(1,length(x_pts)),y_pts(:));    z_pts=z_pts(:);
res_pspec=pseudospectra(G,A,R,z_pts);

%% plot pseudospectral contours
res_pspec_rs=reshape(res_pspec,length(y_pts),length(x_pts));
figure
hold on
box on
v=(10.^(-10:0.3:0));
contourf(reshape(real(z_pts),length(y_pts),length(x_pts)),reshape(imag(z_pts),length(y_pts),length(x_pts)),log10(real(res_pspec_rs)),log10(v));
contourf(reshape(real(z_pts),length(y_pts),length(x_pts)),-reshape(imag(z_pts),length(y_pts),length(x_pts)),log10(real(res_pspec_rs)),log10(v));
cbh=colorbar;
cbh.Ticks=log10(10.^(-2:1:0));
cbh.TickLabels=10.^(-2:1:0);
clim([-2,0]);
reset(gcf)
set(gca,'YDir','normal')
colormap(inferno(100))

title('Pseudospectrum of Duffing oscillator','interpreter','latex','fontsize',18)
xlabel('$\mathrm{Re}(z)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{Im}(z)$','interpreter','latex','fontsize',18)

plot(sin(0:0.01:2*pi), cos(0:0.01:2*pi),'--','color','white');

ax=gca; ax.FontSize=18;
axis equal; axis([-1.2 1.2 -1.2 1.2])
exportgraphics(gcf,'duffing_oscillator_pspec.pdf','ContentType','vector','BackgroundColor','none')


%% define kernel
function ker=kernel(x,t)
    %d=2, use matern kernel with n=3
    r=vecnorm(x-t);
    r=r*6;
    ker=zeros(1,size(x,2));
    ker(r>0)=(r(r>0)).^(2).*besselk(-2,r(r>0));
    ker(r==0)=2;
end

%% define duffing oscillator
function dxdt = duffing(~,x)
    dxdt=zeros(2,1);
    dxdt(1)=x(2);
    dxdt(2)=-0.2*x(2)-x(1)-x(1).^3;
end
