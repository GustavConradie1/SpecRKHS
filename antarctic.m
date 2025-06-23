%% Koopman/PF analysis of antarctic sea ice concentration data
clear
addpath(genpath('./data/antarctic_data'))
addpath(genpath('./algorithms'))
addpath(genpath('./colormaps'))
rng(0)

%% load data sets
load('DATA_antarctic.mat')
load('LAND_antarctic.mat')
load('nLAND_antarctic.mat')

[d,N]=size(DATA);
steps=60;
lag=10; %remove final 10 months from dataset so we go from beginning of 1979 to end of 2023 - 45 years of data
max=(N-steps-lag); %use all but last 5 years as training data
min=1;
x=DATA(:,min:max-1);
y=DATA(:,min+1:max);

%% compute matrices
ker=@(x,t) kernel_matern(x,t);
[G,A,R]=generate_matrices_kernelized(x,y,ker);

%% compute verified eigenvalues
[Lambda_res,F_res,Lambda,F,res,res_verif,idx,W,W_res]=verified_eigenvalues(G,A,R,11);
length(idx)

%% plot spurious and verified eigenvalues
figure
scatter(angle(Lambda),log(abs(Lambda)),200,res,'.','LineWidth',1);
hold on
scatter(angle(Lambda_res),log(abs(Lambda_res)),500,res_verif,'.','LineWidth',1);
box on
clim([0,0.04])
load('cmap.mat')
colormap(cmap2); colorbar
xlabel('$\mathrm{arg}(\lambda)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{log}(|\lambda|)$','interpreter','latex','fontsize',18)
title(['Residuals for Antarctic sea ice data',newline],'interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18; axis([-pi pi -20*10^(-3) 10^(-3)])
for k=-5:1:5
    plot(k*pi/6*ones(22,1),-20*10^(-3):10^(-3):10^(-3),'--','Color','black')
end
xticks([-pi -5*pi/6 -4*pi/6 -3*pi/6 -2*pi/6 -pi/6 0 pi/6 2*pi/6 3*pi/6 4*pi/6 5*pi/6 pi])
set(groot,'defaultAxesTickLabelInterpreter','latex');  
xticklabels({'$-\pi$','','$-2\pi/3$','','$-\pi/3$','','$0$','','$\pi/3$','','$2\pi/3$','','$\pi$'})
xtickangle(30)
exportgraphics(gcf,'antarctic_evals_angle.pdf','ContentType','vector','BackgroundColor','none')

%% compute Perron-Frobenius modes and plot
for j=[1 4 6]
    idx=j;
    L=Lambda_res(idx); F=F_res(:,idx); r=res_verif(idx);
    Phi=((G*F)\(x.')).';
    figure
    u = Phi;
    u = real(u*exp(1i*mean(angle(u))));
    u=-u;
    v = zeros(332*316,1)+NaN;
    v(nLAND) = u(:);
    v = reshape(v,[332,316]);
    imagesc(v,'AlphaData',~isnan(v))
    colormap(coolwarm)
    clim([mean(u(:))-2*std(u(:)) mean(u(:))+2*std(u(:))])
    set(gca,'Color',[1,1,1]*0.6)
    axis equal
    axis tight
    box on
    set(gca,'xticklabel',{[]})
    set(gca,'yticklabel',{[]})
    hold on
    plot(50*sin(0:0.01:2*pi)+317/2, 50*cos(0:0.01:2*pi)+333/2,'color','white');
    plot(100*sin(0:0.01:2*pi)+317/2, 100*cos(0:0.01:2*pi)+333/2,'color','white');
    plot(150*sin(0:0.01:2*pi)+317/2, 150*cos(0:0.01:2*pi)+333/2,'color','white');
    plot(ones(332,1)*317/2,(1:1:332),'color','white')
    plot(1:1:316,(1/2)*(1:1:316)-317/4+333/2,'color','white')
    plot(1:1:316,-(1/2)*(1:1:316)+317/4+333/2,'color','white')
    text(95,190,{'West','Antarctica'},'interpreter','latex','color','black')
    text(180,130,{'East','Antarctica'},'interpreter','latex','color','black')
    text(5,95,{'South','America'},'interpreter','latex','color','black')
    title(['$\lambda' sprintf('=$ %g+%hi, residual $=$ %f',real(L),imag(L),r)],'interpreter','latex','fontsize',18)
    exportgraphics(gcf,sprintf('antarctic_evals_mode_%d.pdf',j),'ContentType','vector','BackgroundColor','none','resolution',1000)
end

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
grid on

title('Pseudospectrum of Antarctic sea ice data','interpreter','latex','fontsize',18)
xlabel('$\mathrm{Re}(z)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{Im}(z)$','interpreter','latex','fontsize',18)

ax=gca; ax.FontSize=18; axis equal tight;   axis([x_pts(1),x_pts(end),-y_pts(end),y_pts(end)])
exportgraphics(gcf,'antarctic_pspec.pdf','ContentType','vector','BackgroundColor','none')


%% use SpecRKHS-Obs to predict future values of sea ice concentration (to be compared to 5 years of test data)
x0=DATA(:,max);
Kx0_vals=zeros(max-min,1);
for i=1:max-min
    Kx0_vals(i)=ker(x0,x(:,i));
end
coefs_res=((G*F_res)\Kx0_vals).';
x_kmd=real((conj(coefs_res).*(conj(Lambda_res).^(1:steps)).')*(F_res'*x(1:d,:).')).';

%% do the same but using kEDMD
G_start=zeros(1,max-min);
for i=1:max-min
    G_start(i)=ker(x0,x(:,i));
end
mode_full=(([G; G_start]*W)\(DATA(:,min:max).')).';
psi0_full=G_start*W;
x_kedmd=real(transpose(transpose(psi0_full).*(conj(Lambda).^(1:steps)))*mode_full.')';

%% Compare to DMD

[U,S,~] = svd(x,'econ');
r = rank(S);
U = U(:,1:r);
            
PXs = x'*U;
PYs = y'*U;
K = PXs\PYs;
[W1,LAM,W2] = eig(K,'vector');
PXr = PXs*W1; PYr = PYs*W1;

c = ([PXr(1,:);PYr])\transpose([x y(:,end)]);
x_dmd = real(transpose(transpose(PYr(end,:)).*(LAM.^(1:steps)))*c)';


%% Plot relative forecast errors compared to test data
real_data=DATA(:,N-steps-lag+(1:steps));

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
exportgraphics(gcf,'antarctic_forecast_error_delay.pdf','ContentType','vector','BackgroundColor','none')

%% Produce exact sea ice concentration map for a period of 3 months
for k=22:24
    figure
    u = DATA(:,max+k);
    v = zeros(332*316,1)+NaN;
    v(nLAND) = u(:);
    v = reshape(v,[332,316]);
    imagesc(v,'AlphaData',~isnan(v))
    colormap(coolwarm)
    colorbar
    clim([0 100])
    set(gca,'Color',[1,1,1]*0.6)
    axis equal
    axis tight
    set(gca,'xticklabel',{[]})
    set(gca,'yticklabel',{[]})
    title(sprintf('Exact, month %d',k),'fontsize',18,'interpreter','latex')
    exportgraphics(gcf,sprintf('antarctic_exact%d.pdf',k),'ContentType','vector','BackgroundColor','none')
end

%% Produce predicted sea ice concentration map for a period of 3 months 
for k=22:24
    figure
    u = real(x_kmd(:,k).');
    v = zeros(332*316,1)+NaN;
    v(nLAND) = u(:);
    v = reshape(v,[332,316]);
    imagesc(v,'AlphaData',~isnan(v))
    colormap(coolwarm)
    colorbar
    clim([0 100])
    set(gca,'Color',[1,1,1]*0.6)
    axis equal
    axis tight
    set(gca,'xticklabel',{[]})
    set(gca,'yticklabel',{[]})
    title(sprintf('Predicted, month %d',k),'fontsize',18,'interpreter','latex')
    exportgraphics(gcf,sprintf('antarctic_predict%d.pdf',k),'ContentType','vector','BackgroundColor','none')
end

%% matern kernel n=41454, n-d/2=3/2
function ker=kernel_matern(x,t)
    sigma=1/20000;
    r=vecnorm(x-t);
    ker=zeros(1,size(x,2));
    ker(r>0)=(sigma*r(r>0)).^(3/2).*besselk(-3/2,sigma*r(r>0));
    ker(r==0)=sqrt(pi/2);
end