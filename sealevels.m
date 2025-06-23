%% Koopman/PF analysis of Northern hemisphere sea level heights

clear
addpath(genpath('./data/sealevel_data'))
addpath(genpath('./algorithms'))
addpath(genpath('./colormaps'))
rng(0)

%% load data sets
load('DATA_north.mat')
load('LAND_north.mat')
load('nLAND_north.mat')
DATA=DATA_north;
LAND=LAND_north;
nLAND=nLAND_north;

%% separate into training and test data sets
[d,N]=size(DATA);
steps=60;
maxx=(N-steps); %use all but last 5 years as training data
min=1;
x=DATA(:,min:maxx-1);
y=DATA(:,min+1:maxx);

%% compute matrices
ker=@(x,t) kernel_matern(x,t);
[G,A,R]=generate_matrices_kernelized(x,y,ker);

%% compute verified eigenvalues
[Lambda_res,F_res,Lambda,F,res,res_verif,idx,W,W_res]=verified_eigenvalues(G,A,R,5);
length(idx)

%% compute Perron-Frobenius modes and plot
for idx=[2 4]
    L=Lambda_res(idx); F=F_res(:,idx); r=res_verif(idx);
    Phi=((G*F)\(x.')).';
    figure
    u = Phi;
    u = real(u*exp(1i*mean(angle(u))));
    v = zeros(150*720,1)+NaN;
    v(nLAND) = u(:);
    v = reshape(v,[720,150]);
    v=flip(v.');
    imagesc(v,'AlphaData',~isnan(v))
    colormap(coolwarm)
    clim([mean(u(:))-2*std(u(:)) mean(u(:))+2*std(u(:))])
    set(gca,'Color',[1,1,1]*0.6)
    axis equal
    axis tight
    grid on
    box on
    set(gca,'xticklabel',{[]})
    set(gca,'yticklabel',{[]})
    title(sprintf('Residual$=$%f',r),'interpreter','latex','fontsize',10)
    exportgraphics(gcf,sprintf('sea_levels_evals_mode_%d.pdf',idx),'ContentType','vector','BackgroundColor','none')
end

%% plot spurious and verified eigenvalues
figure
scatter(angle(Lambda),log(abs(Lambda)),200,res,'.','LineWidth',1);
hold on
scatter(angle(Lambda_res),log(abs(Lambda_res)),500,res_verif,'.','LineWidth',1);
box on
clim([0,0.1])
load('cmap.mat')
colormap(cmap2); colorbar
xlabel('$\mathrm{arg}(\lambda)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{log}(|\lambda|)$','interpreter','latex','fontsize',18)
title(['Residuals for sea level data',newline],'interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18; axis([-pi pi -20*10^(-3) 10^(-3)])
for k=-5:1:5
    plot(k*pi/6*ones(22,1),-20*10^(-3):10^(-3):10^(-3),'--','Color','black')
end
xticks([-pi -5*pi/6 -4*pi/6 -3*pi/6 -2*pi/6 -pi/6 0 pi/6 2*pi/6 3*pi/6 4*pi/6 5*pi/6 pi])
set(groot,'defaultAxesTickLabelInterpreter','latex');  
xticklabels({'$-\pi$','','$-2\pi/3$','','$-\pi/3$','','$0$','','$\pi/3$','','$2\pi/3$','','$\pi$'})
xtickangle(30)
exportgraphics(gcf,'sea_level_evals_angle.pdf','ContentType','vector','BackgroundColor','none')

%% compute psuedospectra
pts=100;
x_pts=linspace(-1.2,1.2,pts);    y_pts=linspace(-0.02,1.2,pts/2);
z_pts=kron(x_pts,ones(length(y_pts),1))+1i*kron(ones(1,length(x_pts)),y_pts(:));    z_pts=z_pts(:);
res_pspec=pseudospectra(G,A,R,z_pts);

%% plot pseudospectral contours
res_pspec_rs=reshape(res_pspec,length(y_pts),length(x_pts));
figure
hold on
box on
v=(10.^(-10:0.2:0));
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

plot(sin(0:0.01:2*pi), cos(0:0.01:2*pi),'--','color','white'); %plot unit circle

title('Pseudospectrum for sea level data','interpreter','latex','fontsize',18)
xlabel('$\mathrm{Re}(z)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{Im}(z)$','interpreter','latex','fontsize',18)

ax=gca; ax.FontSize=18; axis equal tight;   axis([x_pts(1),x_pts(end),-y_pts(end),y_pts(end)])
exportgraphics(gcf,'sea_level_pspec.pdf','ContentType','vector','BackgroundColor','none')

%% compute predictions using SpecRKHS-Obs
x0=DATA(:,maxx);
Kx0_vals=zeros(maxx-min,1);
for i=1:maxx-1
    Kx0_vals(i)=ker(x0,x(:,i));
end
coefs_res=((G*F_res)\Kx0_vals).';
x_kmd=real((coefs_res.*((Lambda_res).^(1:steps)).')*(F_res.'*x.')).';

%% kEDMD predictions instead using KMD
G_start=zeros(1,maxx-min);
for i=1:maxx-min
    G_start(i)=ker(x0,x(:,i));
end
mode_full=(([G; G_start]*W)\(DATA(:,min:maxx).')).';
psi0_full=G_start*W;
x_kedmd=real(transpose(transpose(psi0_full).*(conj(Lambda).^(1:steps)))*mode_full.')';

%% Compare to DMD
[U,S,~] = svd(x,'econ');
r = rank(S);
U = U(:,1:r);
PXs = x'*U;
PYs = y'*U;
K = PXs\PYs;
[W,LAM,W2] = eig(K,'vector');
PXr = PXs*W; PYr = PYs*W;
c = ([PXr(1,:);PYr])\transpose([x y(:,end)]);
x_dmd = real(transpose(transpose(PYr(end,:)).*(LAM.^(1:steps)))*c)';

%% Plot relative forecast errors
real_data=DATA(:,N-steps+(1:steps));

er1 = sum(abs(x_kmd-real_data).^2,1)./sum(abs(real_data).^2,1);
er2 = sum(abs(x_dmd-real_data).^2,1)./sum(abs(real_data).^2,1);
er3 = sum(abs(x_kedmd-real_data).^2,1)./sum(abs(real_data).^2,1);
figure
plot(er2,'linewidth',2)
hold on
plot(er3,'linewidth',2)
plot(er1,'linewidth',2)

grid on

title('Relative forecast errors comparison','fontsize',18,'interpreter','latex')
xlabel('Lead Time (Months)','interpreter','latex','fontsize',18)
ylabel('Relative Forecast Error','interpreter','latex','fontsize',18)
legend({'DMD','kEDMD','SpecRKHS-Obs'},'interpreter','latex','fontsize',16,'location','best')
exportgraphics(gcf,'sea_level_error_delay.pdf','ContentType','vector','BackgroundColor','none')

%% use SpecRKHS-Obs for mean sea level observable
mean_sea_level_old=zeros(maxx-1,1);
for j=1:maxx-1
    mean_sea_level_old(j)=mean(DATA(:,j));
end
mode_res=((G*F_res)\(mean_sea_level_old));
mean_sea_level_kmd=real((coefs_res.*((Lambda_res).^(0:steps)).')*(F_res.'*G*F_res*mode_res)).';

%% same computation but using kEDMD
mean_sea_level_old_full=zeros(maxx,1);
for j=1:maxx
    mean_sea_level_old_full(j)=mean(DATA(:,j));
end
mode_full=(([G; G_start]*W)\(mean_sea_level_old_full)).';
psi0_full=G_start*W;
mean_sea_level_kedmd=real(transpose(transpose(psi0_full).*(conj(Lambda).^(0:steps)))*mode_full.')';

%% exact values of mean sea level observable
G_future=zeros(steps+1,maxx-min);
for i=1:steps+1
    for j=1:maxx-min
        G_future(i,j)=ker(DATA(:,maxx+i-1),DATA(:,j));
    end
end
mean_sea_level_observable_exact=G_future*F_res*mode_res;

%% exact values of mean sea level
mean_sea_level_exact=zeros(steps+1,1);
for j=0:steps
    mean_sea_level_exact(j+1)=mean(DATA(:,N-steps+j));
end

%% compute error in approximating K_{x_0}
Kx0_future_vals=zeros(steps+1,1);
for i=0:steps
    Kx0_future_vals(i+1)=ker(x0,DATA(:,maxx+i));
end
Kx0_predict=real((coefs_res.*((Lambda_res).^(0:steps)).')*F_res.'*Kx0_vals);

%% plot comparison of mean sea level predictions
figure
p1=plot(0:1:steps,mean_sea_level_kedmd,'linewidth',2,'color',[0.8500    0.3250    0.0980]);
hold on
p2=plot(0:1:steps,mean_sea_level_kmd,'linewidth',2,'color',[0.9290    0.6940    0.1250]);
p3=plot(0:1:steps,mean_sea_level_exact,'--','linewidth',2,'color',[0    0.4470    0.7410]);

grid on

title('Mean sea level forecast comparison','fontsize',18,'interpreter','latex')
xlabel('Lead Time (Months)','interpreter','latex','fontsize',18)
ylabel('Mean Height Prediction (cm)','interpreter','latex','fontsize',18)
legend([p3 p1 p2],{'Exact','kEDMD','SpecRKHS-Obs'},'interpreter','latex','fontsize',16,'location','northeast')
exportgraphics(gcf,'sea_level_prediction_mean.pdf','ContentType','vector','BackgroundColor','none')

%% same figure without kEDMD for better visualization
figure
hold on
p2=plot(0:1:steps,mean_sea_level_kmd,'linewidth',2,'color',[0.9290    0.6940    0.1250]);
p3=plot(0:1:steps,mean_sea_level_exact,'linewidth',2,'color',[0    0.4470    0.7410]);
grid on

title('Mean sea level forecast comparison','fontsize',18,'interpreter','latex')
xlabel('Lead Time (Months)','interpreter','latex','fontsize',18)
ylabel('Mean Height Prediction (cm)','interpreter','latex','fontsize',18)
legend([p3 p2],{'Exact','SpecRKHS-Obs'},'interpreter','latex','fontsize',16,'location','southwest')
exportgraphics(gcf,'sea_level_prediction_nokedmd.pdf','ContentType','vector','BackgroundColor','none')

%% plot error between SpecRKHS-Obs prediction and exact values of observable for mean sea level
figure
er = (abs(mean_sea_level_observable_exact-mean_sea_level_kmd.').^2)./(abs(mean_sea_level_observable_exact).^2);
semilogy(0:steps,er,'linewidth',2,'color',[0.9290    0.6940    0.1250])
hold on
title('Mean sea level relative forecast error','fontsize',18,'interpreter','latex')
xlabel('Lead Time (Months)','interpreter','latex','fontsize',18)
ylabel('Relative Forecast Error','interpreter','latex','fontsize',18)
legend({'SpecRKHS-Obs'},'interpreter','latex','fontsize',16,'location','northeast')

%% plot error between SpecRKHS-Obs prediction and exact values of mean sea level
er = (abs(mean_sea_level_exact-mean_sea_level_kmd.').^2)./(abs(mean_sea_level_exact).^2);
figure
semilogy(0:1:steps,er,'linewidth',2,'color',[0.9290    0.6940    0.1250])
title('Mean sea level relative forecast error','fontsize',18,'interpreter','latex')
xlabel('Lead Time (Months)','interpreter','latex','fontsize',18)
ylabel('Relative Forecast Error','interpreter','latex','fontsize',18)
legend({'SpecRKHS-Obs'},'interpreter','latex','fontsize',16,'location','northeast')
exportgraphics(gcf,'sea_level_error_mean.pdf','ContentType','vector','BackgroundColor','none')

%% plot exact mean sea level, exact value of mean sea level observable and SpecRKHS-Obs prediction
figure
hold on
p3=plot(0:1:steps,mean_sea_level_exact,'linewidth',2);
p1=plot(0:1:steps,mean_sea_level_observable_exact,'linewidth',2,'color',[0.4940    0.1840    0.5560]);
p2=plot(0:1:steps,mean_sea_level_kmd,'linewidth',2,'color',[0.9290    0.6940    0.1250]);
grid on

title('Mean sea level forecast comparison','fontsize',18,'interpreter','latex')
xlabel('Lead Time (Months)','interpreter','latex','fontsize',18)
ylabel('Mean Height Prediction (cm)','interpreter','latex','fontsize',18)
legend([p3 p1 p2],{'Exact','Observable','SpecRKHS-Obs'},'interpreter','latex','fontsize',16,'location','southwest')
exportgraphics(gcf,'sea_level_prediction_observable.pdf','ContentType','vector','BackgroundColor','none')

%% plot relative error in predictions of observable function K_{x_0}
figure
er = (abs(Kx0_future_vals-Kx0_predict).^2)./(abs(Kx0_future_vals).^2);
semilogy(0:steps,er,'linewidth',2,'color',[0.9290    0.6940    0.1250])
hold on
title('Relative forecast error for kernel function','fontsize',18,'interpreter','latex')
xlabel('Lead Time (Months)','interpreter','latex','fontsize',18)
ylabel('Relative Forecast Error','interpreter','latex','fontsize',18)
legend({'SpecRKHS-Obs'},'interpreter','latex','fontsize',16,'location','northeast')
exportgraphics(gcf,'sea_level_prediction_kernel_error.pdf','ContentType','vector','BackgroundColor','none')

%% matern kernel d=60330, n-d/2=2
function ker=kernel_matern(x,t)
    sigma=1/10000;
    r=vecnorm(x-t);
    ker=zeros(1,size(x,2));
    ker(r>0)=(sigma*r(r>0)).^2.*besselk(-2,sigma*r(r>0));
    ker(r==0)=2;
end