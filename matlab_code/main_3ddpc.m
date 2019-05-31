%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main_3ddpc recovers 3D refractive index of weakly scattering object. By %
% measuring a few through-focus intensity stacks under partially coherent %
% illuminations, like DPC patterns, 3D phase contrast is captured. Hence, %
% it is possible to retrieve quantitative information of the object via a %
% deconvolution process. With the ADMM algorithm provided in this code,   %
% total variation and positivity constraints can be applied during 3D     %
% refractive index reconstruction. If a GPU device is available, the ADMM %
% iterations will run much faster than using CPU for compuataion.         %
%                                                                         %
%   by Michael Chen                                                       %
%                                                                         %
% Please cite:                                                            %
% M. Chen, L. Tian, and L. Waller, "3D differential phase contrast        %
% microscopy," Biomed. Opt. Express 7, 3940-3950 (2016)                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all;
set(0,'DefaultFigureWindowStyle','docked');
addpath('./3ddpc_functions');

% define global variables and FFT/IFFT operations
global N_x N_y N_z Hr_Hrc Hi_Hrc Hr_Hic Hi_Hic AHI1 AHI2 Dx Dy Dz
F        = @(x) fftn(x);  % 3D FFT operator
IF       = @(x) ifftn(x); % 3D IFFT operator
use_gpu  = false;         % true: use gpu, false: use cpu
datatype = 'single';      % we recommend to save variables with single precision to save RAM

%% Load Data

%LOAD YOUR DATA HERE or USE THE EXAMPLE DATASET (single polystyrene bead)
load('../example_3ddpc_dataset.mat', 'I_DPC'); 
if strcmp(datatype, 'single')==1
    I_DPC   = single(I_DPC);
elseif strsmp(datatype, 'double')==1
    I_DPC   = double(I_DPC);
end
fI_DPC  = zeros(size(I_DPC), 'like', I_DPC);

% mean subtraction and intensity normalization
for stack_idx = 1:size(I_DPC, 4)
    I_load                  = I_DPC(:,:,:,stack_idx);
    I_DPC(:,:,:,stack_idx)  = (I_DPC(:,:,:,stack_idx)-mean(I_load(:)))/mean(I_load(:));
    fI_DPC(:,:,:,stack_idx) = F(I_DPC(:,:,:,stack_idx));
end

clear I_load;
N_x    = size(I_DPC,2);    % number of columns
N_y    = size(I_DPC,1);    % number of rows
N_z    = size(I_DPC,3);    % number of z steps
nangle = size(I_DPC,4);    % number of DPC illumination patterns

% plot the loaded DPC image stacks
for stack_idx = 1:nangle 
    figure('Name', ['DPC image stack, ', num2str(stack_idx)]);
    for plot_idx = 1:3
        subplot(1,3,plot_idx)
        if plot_idx==1
            imagesc(I_DPC(:,:,round(N_z/2)+1, stack_idx));
            title(['DPC_x_y_,_', num2str(stack_idx)], 'fontsize', 24);
            xlabel('x, pixel', 'fontsize', 16);
            ylabel('y, pixel', 'fontsize', 16);
        elseif plot_idx==2
            imagesc(squeeze(I_DPC(:,round(N_x/2)+1,:,stack_idx)));
            title(['DPC_z_y_,_', num2str(stack_idx)], 'fontsize', 24);
            xlabel('z, pixel', 'fontsize', 16);
            ylabel('y, pixel', 'fontsize', 16);
        else
            imagesc(squeeze(I_DPC(round(N_y/2)+1,:,:,stack_idx)));
            title(['DPC_z_x_,_', num2str(stack_idx)], 'fontsize', 24);
            xlabel('z, pixel', 'fontsize', 16);
            ylabel('x, pixel', 'fontsize', 16);
        end
        colormap gray;
        axis square;
        caxis([-0.2, 0.2]);
    end
    drawnow;
end

%% Setup System Parameters

%NEED TO MODIFY THE FOLLOWING PARAMETERS BASED ON EXPERIMENT SETTINGS
sigma           = 1;                      % partial coherence factor
NA_obj          = 0.65;                   % numerical aperture of objective lens
NA_illu         = sigma*NA_obj;           % numerical aperture of illumination
magnification   = 41;                     % magnification of the optical system
rotation        = [0 180 270 90];         % rotation angles of illumination
medium_index    = 1.59;                   % refractive index of immersion media
lambda          = 0.514;                  % wavelength in micron
ps_camera       = 6.5;                    % camera's pixel size in micron
ps              = ps_camera/magnification;% demagnified pixel size in micron
psz             = 1.0;                    % z step in micron

%% Setup System Coordinates

% coordinates in image space
x       = gen1DCoordinate(N_x, ps, datatype);
y       = gen1DCoordinate(N_y, ps, datatype);
z       = gen1DCoordinate(N_z, psz, datatype);
[X,Y,Z] = meshgrid(x, y, z);
z       = ifftshift(z);

% coordinates in frequency space
dfx     = 1/N_x/ps;
dfy     = 1/N_y/ps;
dfz     = 1/N_z/psz;
fx      = gen1DCoordinate(N_x, dfx, datatype);
fy      = gen1DCoordinate(N_y, dfy, datatype);
fz      = gen1DCoordinate(N_z, dfz, datatype);
[Fx,Fy] = meshgrid(fx, fy);
[~,~,Fz]= meshgrid(fx, fy, fz);
Fx      = ifftshift(Fx);
Fy      = ifftshift(Fy);
Fz      = ifftshift(Fz);

%% Compute Transferfunctions

% if image has lateral dimensions smaller than 512 pixels, first upsample
% the source and pupil for transfer function calculation and then
% downsample to the original image size.
if N_x < 512
   downsamp_rate_col = ceil(512/N_x);
   upsamp_N_x        = downsamp_rate_col*N_x;
else
   downsamp_rate_col = 1;
   upsamp_N_x        = N_x;
end

if N_y < 512
   downsamp_rate_row = ceil(512/N_y);
   upsamp_N_y        = downsamp_rate_row*N_y;
else
   downsamp_rate_row = 1;
   upsamp_N_y        = N_y;
end

% generate upsampled coordinates for transfer functions evaluation
upsamp_dfx                 = 1/upsamp_N_x/ps;
upsamp_dfy                 = 1/upsamp_N_y/ps;
upsamp_fx                  = gen1DCoordinate(upsamp_N_x, upsamp_dfx, datatype);
upsamp_fy                  = gen1DCoordinate(upsamp_N_y, upsamp_dfy, datatype);
upsamp_fx                  = ifftshift(upsamp_fx);
upsamp_fy                  = ifftshift(upsamp_fy);
[upsamp_Fx,upsamp_Fy]      = meshgrid(upsamp_fx, upsamp_fy);

% allocate memories for transfer functions
sources                    = zeros(upsamp_N_y, upsamp_N_x, nangle, 'like', fI_DPC);
pupil                      = ((upsamp_Fx.^2+upsamp_Fy.^2)*lambda^2<NA_obj^2);
imaginaryTransferFunction  = zeros(size(fI_DPC), 'like', fI_DPC);
realTransferFunction       = zeros(size(fI_DPC), 'like', fI_DPC);

% compute 3D transfer functions for individual DPC intensity stack
for stackIdx = 1:nangle
    sources(:,:,stackIdx)  = sourceCompute(rotation(stackIdx), NA_illu, lambda, upsamp_Fx, upsamp_Fy);
    sources(:,:,stackIdx)  = fftshift(sources(:,:,stackIdx));
    [ImT,ReT]              = genTransferFunction3D(lambda,sources(:,:,stackIdx),pupil,z,upsamp_Fx,upsamp_Fy,medium_index);
    imaginaryTransferFunction(:,:,:,stackIdx) = permute(downsample(permute(downsample(ImT,downsamp_rate_row),[2,1,3])...
                                                                    ,downsamp_rate_col),[2,1,3]); 
    realTransferFunction(:,:,:,stackIdx)      = permute(downsample(permute(downsample(ReT,downsamp_rate_row),[2,1,3])...
                                                                    ,downsamp_rate_col),[2,1,3]);
end
    
clear ReT ImT

ReT_display = fftshift(realTransferFunction(:,:,:,1));
figure('Name', '3D Phase Transfer Function of the first DPC pattern');
subplot(1,3,1)
imagesc(fx, fy, squeeze(imag(ReT_display(:,:,N_z/2+1))),[-.12,.12]);colormap jet;axis square;
title('fx-fy','fontsize',24);
xlabel('fx, \mu m^-^1', 'fontsize', 16);
ylabel('fy, \mu m^-^1', 'fontsize', 16);
subplot(1,3,2)
imagesc(fz, fy, squeeze(imag(ReT_display(:,N_x/2+1,:))),[-.12,.12]);colormap jet;axis square;
title('fz-fy','fontsize',24);
xlabel('fz, \mu m^-^1', 'fontsize', 16);
ylabel('fy, \mu m^-^1', 'fontsize', 16);
subplot(1,3,3)
imagesc(fz, fx, squeeze(imag(ReT_display(N_y/2+1,:,:))),[-.12,.12]);colormap jet;axis square;
title('fz-fx','fontsize',24);
xlabel('fz, \mu m^-^1', 'fontsize', 16);
ylabel('fx, \mu m^-^1', 'fontsize', 16);
drawnow;

%% Precalculation

% gradient operations
Dx          = zeros(N_y,N_x,N_z,datatype); Dx(1,1,1) = 1; Dx(1,end,1) = -1;
Dx          = F(Dx);
Dy          = zeros(N_y,N_x,N_z,datatype); Dy(1,1,1) = 1; Dy(end,1,1) = -1;
Dy          = F(Dy);
Dz          = zeros(N_y,N_x,N_z,datatype); Dz(1,1,1) = 1; Dz(1,1,end) = -1;
Dz          = F(Dz);

% coefficients from the transfer functions
Hr_Hrc      = sum(abs(realTransferFunction).^2,4);
Hi_Hrc      = sum(imaginaryTransferFunction.*conj(realTransferFunction),4);
Hr_Hic      = sum(realTransferFunction.*conj(imaginaryTransferFunction),4);
Hi_Hic      = sum(abs(imaginaryTransferFunction).^2,4);
AHI1        = sum(fI_DPC.*conj(realTransferFunction),4);
AHI2        = sum(fI_DPC.*conj(imaginaryTransferFunction),4);
modified    = false; %a flag to see if the above values have been modified

%% Start The Iterative Algorithm (ADMM) or Tikhonov Regularization
method        = 'Tikhonov';       %'TV' or 'Tikhonov'
max_iteration = 50;         % maximum iteration for ADMM
rho           = 4e-5;       % penalty parameter
tau           = 8e-5;       % total variation regularization parameter
SP1_k         = zeros(N_y, N_x, N_z, 2, datatype); % scattering potential
SP2_k         = zeros(N_y, N_x, N_z, 2, datatype); % splitting of scattering potential 
DSP_k         = zeros(N_y, N_x, N_z, 6, datatype); % 3D gradient vectors
y1_k          = zeros(N_y, N_x, N_z, 6, datatype); % Lagrange multipliers
y2_k          = zeros(N_y, N_x, N_z, 2, datatype); % Lagrange multipliers

if strcmp(method, 'Tikhonov')
    fprintf('Solving 3D DPC with Tikhonov regularization\n');
    tic;
    if ~modified
        Hr_Hrc    = Hr_Hrc + rho;
        Hi_Hic    = Hi_Hic + rho;
        modified  = true; 
    end
    if use_gpu
        Hr_Hrc    = gpuArray(Hr_Hrc);
        Hi_Hrc    = gpuArray(Hi_Hrc);
        Hr_Hic    = gpuArray(Hr_Hic);
        Hi_Hic    = gpuArray(Hi_Hic);
        AHI1      = gpuArray(AHI1);
        AHI2      = gpuArray(AHI2);
    end
    SP1_k   = optimize_SP1(SP2_k, method, use_gpu);
    fprintf('elapsed time: %5.2f seconds\n', toc());
elseif strcmp(method, 'TV')
    fprintf('Solving 3D DPC with total variation regularization and boundary value constraint\n');
    tic;
    if ~modified
        Hr_Hrc    = Hr_Hrc + rho * ( abs(Dx).^2 + abs(Dy).^2 + abs(Dz).^2 + 1);
        Hi_Hic    = Hi_Hic + rho * ( abs(Dx).^2 + abs(Dy).^2 + abs(Dz).^2 + 1);
        modified  = true;
    end
    if use_gpu
        SP1_k     = gpuArray(SP1_k);
        SP2_k     = gpuArray(SP2_k);
        DSP_k     = gpuArray(DSP_k);
        y1_k      = gpuArray(y1_k);
        y2_k      = gpuArray(y2_k);
        Dx        = gpuArray(Dx);
        Dy        = gpuArray(Dy);
        Dz        = gpuArray(Dz);
        Hr_Hrc    = gpuArray(Hr_Hrc);
        Hi_Hrc    = gpuArray(Hi_Hrc);
        Hr_Hic    = gpuArray(Hr_Hic);
        Hi_Hic    = gpuArray(Hi_Hic);
        AHI1      = gpuArray(AHI1);
        AHI2      = gpuArray(AHI2);
    end

    figure('Name', 'Recovered 3D RI Over Iterations');
    for iter = 1:max_iteration

        % solve Least-Squares
        SP1_k            = optimize_SP1(SP2_k,  method, use_gpu, DSP_k, y1_k, y2_k, rho);

        % solve LASSO proximal step
        [DSP_k,DSP_k1_k] = prox_L1(SP1_k, y1_k, rho, tau, use_gpu);

        % solve Euclidean proximal step
        SP2_k            = prox_projection(SP1_k, y2_k, [-1,-1]);

        % dual update
        y1_k             = y1_k + DSP_k1_k;
        y2_k             = y2_k + (SP1_k - SP2_k);

        % convert scattering potential to RI
        RI_3D            = convertScatteringPotentialToRI(SP2_k, lambda, medium_index);

        hold on;
        subplot(1,3,1)
        imagesc(x,y,RI_3D(:,:,N_z/2+1,1));
        caxis([medium_index-0.01,medium_index+0.01]);
        axis([-15, 15, -15, 15]);
        axis square; colormap jet;
        xlabel('x, \mum', 'fontsize', 16)
        ylabel('y, \mum', 'fontsize', 16)
        title('RI_x_y','fontsize',24);

        subplot(1,3,2)
        imagesc(fftshift(z),y,squeeze(RI_3D(:,round(N_x/2),:,1)));
        caxis([medium_index-0.01,medium_index+0.01]);
        axis([-15, 15, -15, 15]);
        axis square; colormap jet;
        xlabel('z, \mum', 'fontsize', 16)
        ylabel('y, \mum', 'fontsize', 16)
        title('RI_z_y','fontsize',24);

        subplot(1,3,3)
        imagesc(fftshift(z),x,squeeze(RI_3D(round(N_y/2),:,:,1))); 
        caxis([medium_index-0.01,medium_index+0.01]);
        axis([-15, 15, -15, 15]);
        axis square; colormap jet;
        xlabel('z, \mum', 'fontsize', 16)
        ylabel('x, \mum', 'fontsize', 16)
        title('RI_z_x','fontsize',24);
        drawnow;
        
        if iter>1
            fprintf(repmat('\b',1,43+(floor(log10(max_iteration))+1)*2));
        end
        fprintf('elapsed time: %5.2f seconds, iteration : %02d/%02d\n', toc(), iter, max_iteration);
    end
    close('Name', 'Recovered 3D RI Over Iterations');
end
%% Display The Optimized 3D RI

RI_3D = convertScatteringPotentialToRI(SP1_k, lambda, medium_index);

figure('Name', 'Final 3D RI Reconstruction');
subplot(1,3,1)
imagesc(x,y,RI_3D(:,:,N_z/2+1)); axis square; colormap jet;
caxis([medium_index-0.01,medium_index+0.01]);
axis([-15, 15, -15, 15]); 
xlabel('x, \mum', 'fontsize', 16)
ylabel('y, \mum', 'fontsize', 16)
title('RI_x_y','fontsize',24);

subplot(1,3,2)
imagesc(fftshift(z),y,squeeze(RI_3D(:,round(N_x/2),:,1))); axis square; colormap jet;
caxis([medium_index-0.01,medium_index+0.01]);
axis([-15, 15, -15, 15]);
xlabel('z, \mum', 'fontsize', 16)
ylabel('y, \mum', 'fontsize', 16)
title('RI_z_y','fontsize',24);

subplot(1,3,3)
imagesc(fftshift(z),x,squeeze(RI_3D(round(N_y/2),:,:,1))); axis square; colormap jet;
caxis([medium_index-0.01,medium_index+0.01]);
axis([-15, 15, -15, 15]);
xlabel('z, \mum', 'fontsize', 16)
ylabel('x, \mum', 'fontsize', 16)
title('RI_z_x','fontsize',24);
drawnow;
