%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%prox_projection performs Euclidean norm projection to impose positivity  %
%or negativity constraints on the scattering potential                    %
%Inputs:                                                                  %
%   SP1_k     : scattering potential                                      %
%   y2_k      : Lagrange multipliers                                      %
%   direction : Euclidean projection directions                           %
%Output:                                                                  %
%   SP2_k     : splitting of scattering potential                         %
%                                                                         %
%   by Michael Chen                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function SP2_k = prox_projection(SP1_k, y2_k, direction)

    SP_re = SP1_k(:,:,:,1) + y2_k(:,:,:,1);
    SP_im = SP1_k(:,:,:,2) + y2_k(:,:,:,2);
    
    if direction(1) == 1
        %positivity constraint on the real part of scattering potential
        SP_re(SP_re<0) = 0;
    else
        %negativity constraint on the real part of scattering potential
        SP_re(SP_re>0) = 0;
    end
    
    if direction(2) == 1
        %positivity constraint on the imaginary part of scattering potential
        SP_im(SP_im<0) = 0;
    else
        %negativity constraint on the imaginary part of scattering potential
        SP_im(SP_im>0) = 0;
    end
    
    SP2_k = cat(4,SP_re,SP_im);
end