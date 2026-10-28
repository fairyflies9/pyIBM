function forcespread(array1, array2, in_f1, in_f2, Q);
%----------------------------------------------------------------
%Written by Laura Miller, 02/12/08. Warning: This code is far from
%optimized or pretty :) Use at your own risk
%
%forcespread spreads the boundary forces to the fluid grid using a discrete delta 
%function. Please see Peskin and McQueen, 1996 for more details.
%
%Inputs
%array1 and array2 hold the cartesian coordinates of the boundary points.
%in_f1 and inf2 hold the boundary forces in Cartesian coordinates.
%Q is the number of boundary points, or the length of the above arrays.
%
%Global variables:
%f1 and f2 are the horizontal and vertical components of the applied force
%on the fluid grid
%dx and dt are the spatial and time steps
%M and N are the number of spatial steps in the x- and y-directions
%
%Matrices modified:
%f1 and f2 are updated
%-----------------------------------------------------------------  

in_fluid; %read in fluid parameters
global f1 f2

for i=1:Q, 
    %Determine which Cartesian gridpoint the boundary point is near.
    x1 = ceil(array1(i,1)/dx);  
    x2 = ceil(array2(i,1)/dx);  
    
    %Spread the force to the nearby neighbors
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
        
        %Now the force is spread using a delta function that is a little
        %different from the one used in Peskin and McQueen.
        f1(ii+1,jj+1) = f1(ii+1,jj+1)+in_f1(i,1)*(1/(4*dx))*(1+cos((pi/(2*dx))*(dx*(x1+1-k)-array1(i,1))))*(1/(4*dx))*(1+cos((pi/(2*dx))*(dx*(x2+1-h)-array2(i,1))))*ds;  
        f2(ii+1,jj+1) = f2(ii+1,jj+1)+in_f2(i,1)*(1/(4*dx))*(1+cos((pi/(2*dx))*(dx*(x1+1-k)-array1(i,1))))*(1/(4*dx))*(1+cos((pi/(2*dx))*(dx*(x2+1-h)-array2(i,1))))*ds;
      end
    end
end
        
