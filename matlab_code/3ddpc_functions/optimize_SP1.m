%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%optimize_SP1 solves the Least-Squares problem in the ADMM iterative      %
%algorithm or finds the analytical solution using Tikhonov regularization %
%Inputs:                                                                  %
%   SP2_k     : splitting of scattering potential                         %
%   method    : ADMM mode or Tikonov mode                                 %
%   use_gpu   : flag to specify gpu usage                                 %
%   DSP_k     : gradient vectors of the scattering potential              %
%   y1_k      : Lagrange multipliers                                      %
%   y2_k      : Lagrange multipliers                                      %
%   rho       : penalty parameter                                         %
%Output:                                                                  %
%   SP1_k     : scattering potential                                      %
%                                                                         %
%   by Michael Chen                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function SP1_k = optimize_SP1(SP2_k, method, use_gpu, DSP_k, y1_k, y2_k, rho)

global Hr_Hrc Hi_Hrc Hr_Hic Hi_Hic AHI1 AHI2 Dx Dy Dz

if nargin<4
    if ~strcmp(method, 'Tikhonov')
        disp('Error in optimize_SP1.m: wrong method flag or more inputs should be provided!');
        return
    end
end

F               = @(x) fftn(x);
IF              = @(x) ifftn(x);
SP1_k           = zeros(size(SP2_k), 'like', SP2_k);
if use_gpu
    SP1_k = gpuArray(SP1_k);
end

denominator     = Hr_Hrc.*Hi_Hic-Hi_Hrc.*Hr_Hic;

if strcmp(method, 'Tikhonov')
    SP1_k(:,:,:,1)  = real(IF((AHI1.*Hi_Hic-AHI2.*Hi_Hrc)./denominator));
    SP1_k(:,:,:,2)  = real(IF((AHI2.*Hr_Hrc-AHI1.*Hr_Hic)./denominator));
elseif strcmp(method, 'TV')
    AHI1_k1         = AHI1 + rho*(F(SP2_k(:,:,:,1) - y2_k(:,:,:,1))...
                       + conj(Dx).*F(DSP_k(:,:,:,1) + y1_k(:,:,:,1))...
                       + conj(Dy).*F(DSP_k(:,:,:,2) + y1_k(:,:,:,2))...
                       + conj(Dz).*F(DSP_k(:,:,:,3) + y1_k(:,:,:,3)));
    AHI2_k1         = AHI2 + rho*(F(SP2_k(:,:,:,2) - y2_k(:,:,:,2))...
                       + conj(Dx).*F(DSP_k(:,:,:,4) + y1_k(:,:,:,4))...
                       + conj(Dy).*F(DSP_k(:,:,:,5) + y1_k(:,:,:,5))...
                       + conj(Dz).*F(DSP_k(:,:,:,6) + y1_k(:,:,:,6)));
    SP1_k(:,:,:,1)  = real(IF((AHI1_k1.*Hi_Hic-AHI2_k1.*Hi_Hrc)./denominator));
    SP1_k(:,:,:,2)  = real(IF((AHI2_k1.*Hr_Hrc-AHI1_k1.*Hr_Hic)./denominator));
end

end