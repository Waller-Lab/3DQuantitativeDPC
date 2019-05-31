%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%convertScatteringPotentialToRI computes the 3D refractive index given a  %
%3D scattering potential                                                  %
%Inputs:                                                                  %
%   scattering_potential: 4D tensors contains real part and imaginary part%
%   wavelength          : wavelength of incident light                    %
%   RI                  : refractive index of the surrounding medium      %
%Output:                                                                  %
%   RI_3D               : computed 3D refractive index                    %
%                                                                         %
%   by Michael Chen                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function RI_3D = convertScatteringPotentialToRI(scattering_potential, wavelength, RI)
    wavenumber  = 2*pi/wavelength;
    B           = -(RI^2-scattering_potential(:,:,:,1)/wavenumber^2);
    C           = -(-scattering_potential(:,:,:,2)/wavenumber^2/2).^2;
    RI_3D       = sqrt((-B+sqrt(B.^2-4*C))/2);
end