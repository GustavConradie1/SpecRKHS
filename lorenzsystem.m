%% Koopman/PF analysis of chaotic Lorenz system
clear

addpath(genpath('./algorithms'))
addpath(genpath('./colormaps'))

rng(1)

%% generate snapshot data
time_step=0.01;
r = 1000; 
no_steps=20;
no_traj=500;
M=no_traj*no_steps;
x0=50*(rand(3,no_traj)-0.5); 
x0(3,:)=x0(3,:)+25;
x=zeros(3,M);
y=zeros(3,M);
for i=1:no_traj
    [~, x1]=ode45(@(t,z) lorenz(t,z),0:time_step:(no_steps*time_step),x0(:,i));
    x1=x1';
    for j=1:no_steps
        x(:,(i-1)*no_steps+j)=x1(:,j);
        y(:,(i-1)*no_steps+j)=x1(:,j+1);
    end
end

%% compute operator folding matrices
ker=@(x,t) kernel(x,t);
[G,A,R]=generate_matrices_kernelized(x,y,ker);
G = (G+G')/2; % helpful since it tells matlab it is hermitian
R = (R+R')/2;

%% compute pseudoeigenfunctions with various angles
angles=(5:5)/10; %(0:5)/10
lambda = exp(1i*pi*angles);

res=zeros(1,length(angles));
pefuns=zeros(M,length(angles));
 
pf = parfor_progress(length(angles));
pfcleanup = onCleanup(@() delete(pf));
for i=1:length(angles)
    [res(i),pefuns(:,i)]=pseudoeigenfunction(G,A,R,lambda(i),r);
    parfor_progress(pf);
end

fun_val = G*pefuns;

%% plot pseudoeigenfunctions along trajectory data
close all
for i=1:length(angles)
    figure
    vals=real(fun_val(:,i));
    vals = vals - mean(vals(:));
    st = std(vals);
    h=scatter3(x(1,:),x(2,:),x(3,:),50,vals,'.','LineWidth',1);
    hold on
    box on
    grid on
    colormap(coolwarm); 
    clim([-st,st]/2)
    colorbar
    xlabel('$u_1$','interpreter','latex','fontsize',18)
    ylabel('$u_2$','interpreter','latex','fontsize',18)
    zlabel('$u_3$','interpreter','latex','fontsize',18)
    ax=gca; ax.FontSize=16; axis equal;
    title(['$\theta' sprintf('=$ %1.2f, residual $=$ %1.3f',angles(i),res(i))],'interpreter','latex','fontsize',18)
    view([36.9,10.9])
    exportgraphics(gcf,sprintf('lorenz_system_pefun_final_%d.pdf',i),'ContentType','vector','BackgroundColor','none')
end

%% compute verified eigenvalues and plot both spurious and verified
[~,~,Lambda,~,res,~,~,~,~]=verified_eigenvalues(G2,A2,R2,0.2);

%% plot spurious and verified eigenvalues
figure
scatter(real(Lambda),imag(Lambda),200,res,'.','LineWidth',1);
hold on
plot(sin(0:0.01:2*pi), cos(0:0.01:2*pi),'-k','LineWidth',1);
box on
clim([0,0.5])
load('cmap.mat')
colormap(cmap2); colorbar
xlabel('$\mathrm{Re}(\lambda)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{Im}(\lambda)$','interpreter','latex','fontsize',18)
title('Residuals of Lorenz system','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18; axis equal; axis([-1.2 1.2 -1.2 1.2])
exportgraphics(gcf,'lorenz_system_evals.pdf','ContentType','vector','BackgroundColor','none')

%% compute psuedospectra
pts=100;
x_pts=linspace(-1.5,1.5,pts);    y_pts=linspace(0,1.5,pts/2);
z_pts=kron(x_pts,ones(length(y_pts),1))+1i*kron(ones(1,length(x_pts)),y_pts(:));    z_pts=z_pts(:);

res_pspec=pseudospectra(G,A,R,z_pts,r);

%% plot pseudospectral contours
res_pspec_rs=reshape(res_pspec,length(y_pts),length(x_pts));
figure
hold on
v=(10.^(-10:0.2:0));
contourf(reshape(real(z_pts),length(y_pts),length(x_pts)),reshape(imag(z_pts),length(y_pts),length(x_pts)),log10(real(res_pspec_rs)),log10(v));
hold on
contourf(reshape(real(z_pts),length(y_pts),length(x_pts)),-reshape(imag(z_pts),length(y_pts),length(x_pts)),log10(real(res_pspec_rs)),log10(v));
cbh=colorbar;
cbh.Ticks=log10(10.^(-2:1:0));
cbh.TickLabels=10.^(-2:1:0);
clim([-2,0]);
reset(gcf)
set(gca,'YDir','normal')
colormap(inferno(100))

title('Pseudospectrum of Lorenz system','interpreter','latex','fontsize',18)
xlabel('$\mathrm{Re}(z)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{Im}(z)$','interpreter','latex','fontsize',18)

plot(sin(0:0.01:2*pi), cos(0:0.01:2*pi),'--','color','white');

ax=gca; ax.FontSize=18; axis equal;   axis([-1.2 1.2 -1.2 1.2])
exportgraphics(gcf,'lorenz_system_pspec.pdf','ContentType','vector','BackgroundColor','none')

%% define kernel
function ker=kernel(x,t)
    %d=3 so we use wendland kernel phi_{3,0} equivalent to H^(2)(R^3)
    r=vecnorm(x-t);
    r=r/10;
    r = min(r,1);
    ker=(1-r).^2;
end

%% define lorenz system
function dxdt = lorenz(~,x)
    sigma=10; rho=28; beta=8/3;
    dxdt=zeros(3,1);
    dxdt(1)=sigma*(x(2)-x(1));
    dxdt(2)=x(1)*(rho-x(3))-x(2);
    dxdt(3)=x(1)*x(2)-beta*x(3);
end