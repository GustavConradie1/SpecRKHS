%% Koopman/PF analysis of turbulent channel flow dataset
clear
addpath(genpath('./data/turbulence_data'))
addpath(genpath('./algorithms'))
addpath(genpath('./colormaps'))
rng(0)

%% load data sets
load('DATA_turb.mat')
[d,N]=size(DATA3d);
x=DATA3d(:,1:N-1);
y=DATA3d(:,2:N);

%% compute matrices
ker=@(x,t) kernel_matern(x,t);
[G,A,R]=generate_matrices_kernelized(x,y,ker);

%% compute verified eigenvalues
[Lambda_res,F_res,Lambda,~,res,res_verif]=verified_eigenvalues(G,A,R,0.02);

%% plot spurious and verified eigenvalues
figure
scatter(angle(Lambda),log(abs(Lambda)),200,res,'.','LineWidth',1);
hold on
box on
clim([0,0.1])
load('cmap.mat')
colormap(cmap2); colorbar
xlabel('$\mathrm{arg}(\lambda)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{log}(|\lambda|)$','interpreter','latex','fontsize',18)
title(['Residuals of turbulent channel flow',newline],'interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18; axis([-pi pi -20*10^(-3) 10^(-3)])
xticks([-pi -5*pi/6 -4*pi/6 -3*pi/6 -2*pi/6 -pi/6 0 pi/6 2*pi/6 3*pi/6 4*pi/6 5*pi/6 pi])
set(groot,'defaultAxesTickLabelInterpreter','latex');  
xticklabels({'$-\pi$','','$-2\pi/3$','','$-\pi/3$','','$0$','','$\pi/3$','','$2\pi/3$','','$\pi$'})
xtickangle(30)
exportgraphics(gcf,'channel3d_evals_angle.pdf','ContentType','vector','BackgroundColor','none')

%% compute psuedospectra
pts=100;
x_pts=linspace(-1.2,1.5,pts);    y_pts=linspace(-0.02,1.2,pts/2);
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

title('Pseudospectrum of turbulent channel flow','interpreter','latex','fontsize',18)
xlabel('$\mathrm{Re}(z)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{Im}(z)$','interpreter','latex','fontsize',18)

ax=gca; ax.FontSize=18; axis equal tight;   axis([x_pts(1),x_pts(end),-y_pts(end),y_pts(end)])
exportgraphics(gcf,'channel3d_pspec.pdf','ContentType','vector','BackgroundColor','none')

%% compute Perron-Frobenius modes
idx=1; %plot for indices 1,5,15
L=Lambda_res(idx); F=F_res(:,idx); r=res_verif(idx);
mode=zeros(d,1);
Phi=((G*F)\(x.')).';

%% plot Perron-Frobenius modes

u = Phi;
u = real(u*exp(1i*mean(angle(u))));

nx = 16; ny = 16; nz = 16;
x_points = linspace(3, 3.3, nx);
y_points = linspace(-0.9, -0.6, ny);
z_points = linspace(0.2, 0.5, nz);

results = reshape(u, [nz, ny, nx]); 

figure

axes1 = axes('FontSize', 18, 'XScale', 'lin', 'YScale', 'lin', 'ZScale', 'lin');
box(axes1, 'on')
hold(axes1,'on')
view(3); 

% Generate 3D contour
[X,Y,Z]=meshgrid(x_points, y_points, z_points);
s =  slice(axes1, X, Y, Z, results, ...
      [x_points(1), x_points(end)], ...
      [y_points(1), y_points(end)], ...
      [z_points(1), z_points(end)]);

shading interp;  

colormap('hot')
colorbar('FontSize', 18);
set(s,'EdgeColor','none')
%clim([-0.7 0.2])


% Title and labels
title(sprintf('Residual$=$%f',r),'interpreter','latex','fontsize',18)
xlabel('$x$','interpreter','latex');
ylabel('$y$','interpreter','latex');
zlabel('$z$','interpreter','latex');
axis tight;

exportgraphics(gcf,'channel_evals_mode.png')

lims=clim;

%% plot slices of Perron-Frobenius modes with fixed y-coordinate

figure
box on 
hold on

% Plotting data
y_coord = 12;
results_2d=zeros(nx,nz);
for i=1:nx
    for j=1:nz
        results_2d(i,j)=results(y_coord,j,i);
    end
end
contourf(x_points, z_points, results_2d, 300, 'LineColor','none','edgecolor','none');
colormap('hot')
colorbar
clim(lims)

% Title and labels
title(sprintf('$y=$%f',round(y_points(y_coord),2)),'interpreter','latex','fontsize',18)
xlabel('$x$','interpreter','latex');
ylabel('$z$','interpreter','latex');
ax=gca; ax.FontSize=18;
axis tight; 
axis equal;
exportgraphics(gcf,'channel_evals_mode_slice.png')

%% matern kernel d=4096, n=2049, n-d/2=1
function ker=kernel_matern(x,t)
    sigma=1/100;
    r=vecnorm(x-t);
    ker=zeros(1,size(x,2));
    ker(r>0)=(sigma*r(r>0)).*besselk(-1,sigma*r(r>0));
    ker(r==0)=1;
end