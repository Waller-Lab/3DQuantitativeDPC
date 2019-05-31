%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%sourceCompute calculates the effective DPC source shape                  %
%Inputs:                                                                  %
%   rot_angle: rotation angle of the asymmetric axis                      %
%   NA_illum : illumination NA                                            %
%   lambda   : wavelength                                                 %
%   Fx,Fy    : spaital frequency axes                                     %
%Output:                                                                  %
%   source   : DPC source shape                                           %
%                                                                         %
%   by Michael Chen                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ source ] = sourceCompute(rot_angle, NA_illum, lambda, Fx, Fy)
% support of the source
source_all  = sqrt(Fx.^2+Fy.^2)<=NA_illum/lambda;
asymmetry   = zeros(size(Fx));

% asymmetric mask based on illumination angle
asymmetry(Fy>=(Fx*tand(rot_angle)))=1;
asymmetry(Fy<=(Fx*tand(rot_angle)))=0;
source = source_all.*asymmetry;

if rot_angle == 270 || rot_angle == 90 || rot_angle == 180
    source = fftshift(source);
    source = padarray(source,[1,1], 'post');
    source = flip(source, 1);
    source = flip(source, 2);
    source = source(1:end-1,1:end-1);
    source = ifftshift(source);
end

end

