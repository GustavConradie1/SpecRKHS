%% example showing slower convergence of EDMD vs kEDMD for Gauss map, as well as spectral pollution
addpath(genpath('./algorithms'))
addpath(genpath('./colormaps'))
rng(0)
clear

%% generate snapshot data
N=201;
sigma = 1;
[x,w]=chebpts(N,[-1,0]); %using chebfun package
y=exp(-2*x.^2)-1-exp(-2);
x=x'; y=y';

%% compute operator folding matrices
ker=@(x,t) kernel(x/sigma,t/sigma);
[G,A,R]=generate_matrices_kernelized(x,y,ker);

%% generate predictions for EDMD and kEDMD and kResDMD
psi=cell(N,1);
for j=1:N
    psi{j}=chebpoly(j-1,[-1 0]); %using chebfun package
end
K=edmd(x,y,w,psi);

%%
psi_rkhs=cell(N,1);
for i=1:N
    psi_rkhs{i}=@(y) kernel(x(i)/sigma,y/sigma);
end
C=function_to_state(x,psi_rkhs);
K_rkhs=(pinv(G)*A')';

steps=100;
x0=-0.05;
z_koopman=zeros(steps+1,N);
z_rkhs=zeros(steps+1,N);
for i=1:N
    z_koopman(1,i)=psi{i}(x0);
    z_rkhs(1,i)=psi_rkhs{i,1}(x0);
end
x_exact=zeros(steps+1,1);
x_exact(1)=x0;
for i=1:steps
    x_exact(i+1)=exp(-2*x_exact(i)^2)-1-exp(-2);
    z_koopman(i+1,:)=K*z_koopman(i,:)';
    z_rkhs(i+1,:)=K_rkhs*z_rkhs(i,:)';
end
x_koopman=(z_koopman(:,2)-1)/2;
x_rkhs=C*z_rkhs';

%% plot comparison of exact, EDMD and kEDMD algorithms
figure
box on
hold on
p1=plot(0:1:steps,x_rkhs,'linewidth',4,'color',[0.9290    0.6940    0.1250]);
p3=plot(0:1:steps,x_koopman,'linewidth',3,'color',[0.8500    0.3250    0.0980]);
p2=plot(0:1:steps,x_exact,'--','linewidth',2,'color',[0    0.4470    0.7410]);
legend([p2 p3 p1],'Exact','EDMD','SpecRKHS-Obs','interpreter','latex','fontsize',18,'location','best')
title('Comparison of predictors for Gauss map','interpreter','latex','fontsize',18)
xlabel('Iterations','interpreter','latex','fontsize',18)
ylabel('$x$','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;

exportgraphics(gcf,'gauss_map_comparison.pdf','ContentType','vector','BackgroundColor','none')

%% compute verified eigenvalues
[~,~,Lambda,~,res]=verified_eigenvalues(G,A,R,0.1);

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
title('Residuals for Gauss map','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18; axis equal
exportgraphics(gcf,'gauss_map_evals.pdf','ContentType','vector','BackgroundColor','none')

%% define kernel
function ker = kernel(x1,x2)
    ker=(x1<=x2).*cosh(x1+1).*cosh(x2)/sinh(1)+(1-(x1<=x2)).*cosh(x1).*cosh(x2+1)/sinh(1);
end