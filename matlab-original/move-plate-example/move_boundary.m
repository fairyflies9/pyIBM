
function [array1, array2] = move_boundary(array1, array2, Q, width);
%----------------------------------------------------------------
%Written by Laura Miller, 02/12/08. Warning: This code is far from
%optimized or pretty :) Use at your own risk
%
%move_boundary moves the boundary at the local fluid velocity using a discrete delta 
%function. Please see Peskin and McQueen, 1996 for more details.
%
%Inputs
%array1 and array2 hold the cartesian coordinates of the boundary points.
%Q is the number of boundary points, or the length of the above arrays.
%
%Global variables:
%dx and dt are the spatial and time steps
%M and N are the number of spatial steps in the x- and y-directions
%u and v are the horizontal and vertical components of the fluid velocity
%
%Matrices modified:
%array1 and array2 are updated (new boundary positions).
%-----------------------------------------------------------------  

in_fluid; %read in fluid parameter values
global u v;

%initial the vector that holds the velocity of each boundary point
uarray=zeros(Q,1);
varray=zeros(Q,1);

%loop through the boundary point and determine the local velocity.
for i = 1:Q,  
    x1 = ceil(array1(i,1)/dx);  
    x2 = ceil(array2(i,1)/dx);  
      
    for k=0:3, 
      for h=0:3,
        ii=x1+1-k;  
        jj=x2+1-h;  
        
        %The following if statements handle what happens if a boundary
        %point is near the edge of the domain. This definitely slows the code down.
        %It is possible to rewrite this for improved speed. Alternative,
        %you can comment it out if the boundary is never near the edge.
        if(ii==-1)
            ii=M-1;
        end
        if(ii==-2)
            ii=M-2;
        end
        if(ii==-3)
            ii=M-3;
        end
        if(ii==M)
            ii=0;
        end
        if(ii==M+1)
            ii=1;
        end
        if(ii==M+2)
            ii=2;
        end
 
        
        if(jj==-1)
            jj=N-1;
        end
        if(jj==-2)
            jj=N-2;
        end
        if(jj==-3)
            jj=N-3;
        end
        
        if(jj==N)
            jj=0;
        end
        if(jj==N+1)
            jj=1;
        end
        if(jj==N+2)
            jj=2;
        end
        %--------------------------------------------
        %Calculate the velocity at the boundary point, take a weighted
        %average of the nearby velocities.
        uarray(i,1) = uarray(i,1)+u(ii+1, jj+1)*(1/(4*dx))*(1+cos((pi/(2*dx))*(dx*(x1+1-k)-array1(i,1))))*(1/(4*dx))*(1+cos((pi/(2*dx))*(dx*(x2+1-h)-array2(i,1))))*dx*dx;  
        varray(i,1) = varray(i,1)+v(ii+1, jj+1)*(1/(4*dx))*(1+cos((pi/(2*dx))*(dx*(x1+1-k)-array1(i,1))))*(1/(4*dx))*(1+cos((pi/(2*dx))*(dx*(x2+1-h)-array2(i,1))))*dx*dx;  
      end
    end
      
    %move the boundary that the local velocity
    array1(i,1) = array1(i,1) + uarray(i,1)*dt;  
    array2(i,1) = array2(i,1) + varray(i,1)*dt;  
    
    %if a boundary point has left the fluid grid, move it to the other
    %side.
    for i=1:Q,
        if(array1(i,1)>width),
            array1(i,1) = array1(i,1)-width;
        end
        
        if(array2(i,1)>width),
            array2(i,1) = array2(i,1)-width;
        end
    end
        
end  
  
  