%% Koopman/PF analysis of MD dataset of intestinal fatty acid binding proteins
clear
addpath(genpath('./data/mdsystem_data'))
addpath(genpath('./algorithms'))
addpath(genpath('./colormaps'))
rng(0)

%% load data sets
load('DATA_md.mat')
[d,N]=size(mDATA);
x=mDATA(:,1:N-1);
y=mDATA(:,2:N);

%% compute matrices
ker=@(x,t) kernel_matern(x,t);
[G,A,R]=generate_matrices_kernelized(x,y,ker);
G=(G+G')/2; %increase stability by forcing self-adjointness
R=(R+R')/2;

%% check 'unitaryness' of matrices
disp(norm(G-R)/norm(G))

%% compute spectral measures
theta=linspace(-pi,pi,500);
rng(0)
g=zeros(N-1,1); %observable orthogonal to constant functions
g(100)=1;
g(200)=-1;
g(300)=1;
g(400)=-1;
specmeas3=unitary_spectral_measure(G,A,g,theta,6,'equi',0.1);
specmeas2=unitary_spectral_measure(G,A,g,theta,6,'equi',0.2);
specmeas1=unitary_spectral_measure(G,A,g,theta,6,'equi',0.5);

%% plot spectral measures
specmeas1=specmeas1/(pi*mean(specmeas1)); %normalization
specmeas2=specmeas2/(pi*mean(specmeas2));
specmeas3=specmeas3/(pi*mean(specmeas3));
figure
plot(theta,specmeas1,'linewidth',2)
hold on
plot(theta,specmeas2,'linewidth',2)
plot(theta,specmeas3,'linewidth',2)
hold off
ylim([0 0.5])
xlim([-pi pi])
xticks([-pi -2*pi/3 -pi/3 0 pi/3 2*pi/3 pi])
set(groot,'defaultAxesTickLabelInterpreter','latex');  
xticklabels({'$-\pi$','$-2\pi/3$','$-\pi/3$','$0$','$\pi/3$','$2\pi/3$','$\pi$'})

title('Spectral measure for MD system','interpreter','latex','fontsize',18)
xlabel('$\theta$','interpreter','latex','fontsize',18)
ylabel('$[K_{\epsilon}*\xi_g](\theta)$','interpreter','latex','fontsize',18)
legend('$\epsilon=0.5$','$\epsilon=0.2$','$\epsilon=0.1$','interpreter','latex','location','southeast')
ax=gca; ax.FontSize=18;
box on
exportgraphics(gcf,'md_spec_meas.pdf','ContentType','vector','BackgroundColor','none')

%% compute verified eigenvalues
[Lambda_res,F_res,Lambda,~,res,res_verif]=verified_eigenvalues(G,A,R,0.01);

%% plot spurious and verified eigenvalues
figure
scatter(angle(Lambda),log(abs(Lambda)),200,res,'.','LineWidth',1);
hold on
box on
clim([0,0.25])
load('cmap.mat')
colormap(cmap2); colorbar
xlabel('$\mathrm{arg}(\lambda)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{log}(|\lambda|)$','interpreter','latex','fontsize',18)
title('Residuals of MD system','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;
xlim([-pi pi])
ylim([-0.05 0])
xticks([-pi -2*pi/3 -pi/3 0 pi/3 2*pi/3 pi])
set(groot,'defaultAxesTickLabelInterpreter','latex');  
xticklabels({'$-\pi$','$-2\pi/3$','$-\pi/3$','$0$','$\pi/3$','$2\pi/3$','$\pi$'})
yticks([-0.05 -0.04 -0.03 -0.02 -0.01 0])
exportgraphics(gcf,'md_evals_angle.pdf','ContentType','vector','BackgroundColor','none')

%% compute psuedospectra
pts=50;
x_pts=linspace(-1.2,1.2,pts);    y_pts=linspace(-0.02,1.2,pts/2);
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
axis equal;
plot(sin(0:0.01:2*pi), cos(0:0.01:2*pi),'--','color','white');

title('Pseudospectrum of MD system','interpreter','latex','fontsize',18)
xlabel('$\mathrm{Re}(z)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{Im}(z)$','interpreter','latex','fontsize',18)

ax=gca; ax.FontSize=18; axis equal tight;   axis([x_pts(1),x_pts(end),-y_pts(end),y_pts(end)])
exportgraphics(gcf,'md_pspec.pdf','ContentType','vector','BackgroundColor','none')

%% matern kernel d=37335, n=18668, n-d/2=1/2 
function ker=kernel_matern(x,t) 
    sigma=1/2000;
    r=vecnorm(x-t);
    ker=zeros(1,size(x,2));
    ker(r>0)=(sigma*r(r>0)).^(1/2).*besselk(-1/2,sigma*r(r>0));
    ker(r==0)=sqrt(pi/2);
end