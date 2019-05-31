%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%prox_L1 performs the proximal operator and solves the LASSO problem for  %
%total variation regularization                                           %
%Inputs:                                                                  %
%   SP1_k     : scattering potential                                      %
%   y1_k      : Lagrange multipliers                                      %
%   y2_k      : Lagrange multipliers                                      %
%   rho       : penalty parameter                                         %
%   tau       : total variation regularization parameter                  %
%   use_gpu   : flag to specify gpu usage                                 %
%Output:                                                                  %
%   DSP_k1    : gradient vectors of the scattering potential              %
%   delta     : change of the gradient vectors                            %
%                                                                         %
%   by Michael Chen                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [DSP_k1, delta] = prox_L1(SP1_k, y1_k, rho, tau, use_gpu)
    global N_x N_y N_z
    
    DSP_k     = zeros(N_y,N_x,N_z,6, 'like', SP1_k);
    if use_gpu
        DSP_k = gpuArray(DSP_k);
    end
    
    DSP_k(:,:,:,1) = SP1_k(:,:,:,1) - circshift(SP1_k(:,:,:,1),[0,-1,0]);
    DSP_k(:,:,:,2) = SP1_k(:,:,:,1) - circshift(SP1_k(:,:,:,1),[-1,0,0]);
    DSP_k(:,:,:,3) = SP1_k(:,:,:,1) - circshift(SP1_k(:,:,:,1),[0,0,-1]);
    DSP_k(:,:,:,4) = SP1_k(:,:,:,2) - circshift(SP1_k(:,:,:,2),[0,-1,0]);
    DSP_k(:,:,:,5) = SP1_k(:,:,:,2) - circshift(SP1_k(:,:,:,2),[-1,0,0]);
    DSP_k(:,:,:,6) = SP1_k(:,:,:,2) - circshift(SP1_k(:,:,:,2),[0,0,-1]);
    
    DSP_k1         = DSP_k - y1_k;  
    DSP_k1         = max(0, DSP_k1 - tau/rho) - max(0, -DSP_k1 - tau/rho);
    delta          = DSP_k1 - DSP_k;
    
end