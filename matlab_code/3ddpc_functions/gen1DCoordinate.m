%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%gen1DCoordinate generates 1D Cartesian coordinate                        %
%Inputs:                                                                  %
%   N         : number of points on the 1D coordinate                     %
%   unit      : separation between points                                 %
%   datatype  : output datatype                                           %
%Output:                                                                  %
%   coordinate: generated 1D coordinate                                   %
%                                                                         %
%   by Michael Chen                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function coordinate = gen1DCoordinate(N, unit, datatype)
    coordinate = -(N-mod(N,2))/2:1:(N-mod(N,2))/2-(mod(N,2)==0);
    coordinate = unit*coordinate;
    if strcmp(datatype, 'single')
        coordinate = single(coordinate);
    elseif strcmp(datatype, 'double')
        coordinate = double(coordinate);
    end
end