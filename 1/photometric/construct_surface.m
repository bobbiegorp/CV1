function [ height_map ] = construct_surface( p, q, path_type )
%CONSTRUCT_SURFACE construct the surface function represented as height_map
%   p : measures value of df / dx
%   q : measures value of df / dy
%   path_type: type of path to construct height_map, either 'column',
%   'row', or 'average'
%   height_map: the reconstructed surface


if nargin == 2
    path_type = 'column';
end

[h, w] = size(p);
height_map_column = zeros(h, w);
height_map_row = zeros(h, w);
%path_type = "row";
%path_type = 'average';

switch path_type
    case 'column'
        % =================================================================
        % YOUR CODE GOES HERE
        % top left corner of height_map is zero
        % for each pixel in the left column of height_map
        %   height_value = previous_height_value + corresponding_q_value
        
        % for each row
        %   for each element of the row except for leftmost
        %       height_value = previous_height_value + corresponding_p_value
        
        for y = 2:h
            height_map_column(y,1) = height_map_column(y-1,1) + q(y,1);
        end
        
        for y = 1:h
            for x = 2:w
                height_map_column(y,x) = height_map_column(y,x-1) + p(y,x);
            end
        end
        height_map = height_map_column;
       
        % =================================================================
               
    case 'row'
        
        % =================================================================
        % YOUR CODE GOES HERE
        for x = 2:w
            height_map_row(1,x) = height_map_row(1,x-1) + p(1,x);
        end
        
        for x = 1:w
            for y = 2:h
                height_map_row(y,x) = height_map_row(y-1,x) + q(y,x);
            end
        end
        height_map = height_map_row;
        % =================================================================
          
    case 'average'
        
        % =================================================================
        % YOUR CODE GOES HERE
        %----column case------
        for y = 2:h
            height_map_column(y,1) = height_map_column(y-1,1) + q(y,1);
        end
        
        for y = 1:h
            for x = 2:w
                height_map_column(y,x) = height_map_column(y,x-1) + p(y,x);
            end
        end
        
        %----rowcase------
        for x = 2:w
            height_map_row(1,x) = height_map_row(1,x-1) + p(1,x);
        end
        
        for x = 1:w
            for y = 2:h
                height_map_row(y,x) = height_map_row(y-1,x) + q(y,x);
            end
        end
        
        %----Averaging them------
        height_map = (height_map_column + height_map_row)/2;
        
        % =================================================================
end


end

