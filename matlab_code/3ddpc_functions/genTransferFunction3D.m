%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%genTransferFunction3D calculates the 3D transfer functions for both real % 
%and imaginary parts of the scattering potential (V)                      %
%Inputs:                                                                  %  
%	lambda : wavelength of illumination                                   %
%	source : source shape                                                 %
%	pupil  : pupil function                                               %
%	z      : position of each layer                                       %
%	Fx     : horizontal coordinate in Fourier space                       %
%	Fy     : vertical coordinate in Fourier space                         %
%   RI     : background medium refractive index                           %
%Outputs:                                                                 %
%   imaginaryTransferFunction: transfer function of imaginary part of V   %    
%   realTransferFunction     : transfer function of real part of V        %
%                                                                         %
%   by Michael Chen                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [imaginaryTransferFunction,realTransferFunction] = genTransferFunction3D(lambda,source,pupil,z,Fx,Fy,RI)
    % 2D FFT operators
    F         = @(x) fft2(x);
    IF        = @(x) ifft2(x);
    
    [N_x, N_y]= size(Fx);
    dfx       = Fx(1,2)-Fx(1,1);
    dfy       = Fy(2,1)-Fy(1,1);
       
    % flip the source
    Sf        = padarray(source,[1,1],'post');
    Sf        = flip(Sf,1); 
    Sf        = flip(Sf,2);
    Sf        = Sf(1:end-1,1:end-1);
    Sf        = ifftshift(Sf);
    
    % evaluate the oblique factor
    G         = 1./sqrt((RI/lambda)^2-(Fx.^2+Fy.^2))/4/pi;
    
    % initialize process bar
    w         = waitbar(0,'Transfer Function Calculating...');
    perc      = 0;
    
    % preallocate memories
    imaginaryTransferFunction = zeros(N_y, N_x, length(z), 'like', source);
    realTransferFunction      = zeros(N_y, N_x, length(z), 'like', source);
    
    for j = 1:length(z)
        porp_phase                       = exp(1i*2*pi*z(j)*sqrt((1/lambda)^2-(Fx.^2+Fy.^2)));
        porp_phase(abs(pupil)==0)        = 0; 
        FPSfph_cFPphG                    = F(Sf.*pupil.*porp_phase).*conj(F(pupil.*porp_phase.*G))*dfx*dfy;
        imaginaryTransferFunction(:,:,j) = 2*IF(real(FPSfph_cFPphG));
        realTransferFunction(:,:,j)      = 2*IF(1i*imag(FPSfph_cFPphG));
    
        if mod(j,round((length(z)-1)/5))==0
           perc = perc+20;
           waitbar(perc/100,w,sprintf('Transfer Function Calculating...%d%%',perc))
        end
    end
    
    window                    = reshape(ifftshift(hamming(length(z))), [1, 1, length(z)]);
    imaginaryTransferFunction = fft(bsxfun(@times,imaginaryTransferFunction,window),[],3)*(z(2)-z(1));
    realTransferFunction      = fft(bsxfun(@times,realTransferFunction,window),[],3)*(z(2)-z(1));
    DC                        = sum(Sf(:).*abs(pupil(:)).^2)*dfx*dfy;
    imaginaryTransferFunction = imaginaryTransferFunction/DC;
    realTransferFunction      = 1i*realTransferFunction/DC;
    close(w);
    
end

