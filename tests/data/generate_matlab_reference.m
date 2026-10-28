function generate_matlab_reference(stiff, nsteps, outfile)
%GENERATE_MATLAB_REFERENCE  Reference data for the pyIBM regression test.
%   Runs Laura Miller's 2008 moving-plate code (matlab-original/move-plate-example)
%   UNCHANGED for nsteps steps with target stiffness `stiff`, no plotting, and saves the
%   state after selected steps. The time loop below is moving_plate.m lines 123-174 with
%   the plotting removed; the setup is lines 14-113 with `clear all`/`clf` removed.
%
%   Example (from this folder, MATLAB R2024a):
%     generate_matlab_reference(1,   100, 'heaving_plate_2008_stiff1.mat')
%     generate_matlab_reference(1e4, 100, 'heaving_plate_2008_stiff1e4.mat')

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', '..', 'matlab-original', 'move-plate-example'));

in_fluid; %parameters for fluid solver

global u v f1 f2 press w1 w2 uft vft w1ft w2ft pft time;
global a b c d theta_1 theta_2;

width = 0.4;
time = 0;
Q = M;

u = zeros(M, N);
v = zeros(M, N);
f1 = zeros(M, N);
f2 = zeros(M, N);
press = zeros(M, N);
w1 = zeros(M, N);
w2 = zeros(M, N);
uft = zeros(M, N);
vft = zeros(M,N);
pft = zeros(M,N);

b1 = zeros(Q,1);
b2 = zeros(Q,1);
bt1 = zeros(Q,1);
bt2 = zeros(Q,1);
fb1 = zeros(Q,1);
fb2 = zeros(Q,1);

%------------------------------------------------------------------------------
% moving_plate.m lines 72-96, verbatim
for i=1:M,  
   for j = 1:N, 
      if(or(and((i==1),(j==1)),and((i==1), (j==(N/2)+1)))),
        a(i,j)=0;  
        b(i,j)=0;  
        c(i,j)=0;  
      elseif(or(and((i==(M/2)+1),(j==1)),and((i==(M/2)+1),(j==(N/2)+1)))),  
        a(i,j)=0;  
        b(i,j)=0;  
        c(i,j)=0;  
      else
        a(i,j)= sin(2*pi*(i-1)/M)*sin(2*pi*(i-1)/M)/((sin(2*pi*(i-1)/M)*sin(2*pi*(i-1)/M)+sin(2*pi*(j-1)/N)*sin(2*pi*(j-1)/N)));  
        b(i,j)= sin(2*pi*(i-1)/M)*sin(2*pi*(j-1)/N)/((sin(2*pi*(i-1)/M)*sin(2*pi*(i-1)/M)+sin(2*pi*(j-1)/N)*sin(2*pi*(j-1)/N)));  
        c(i,j)= sin(2*pi*(j-1)/N)*sin(2*pi*(j-1)/N)/((sin(2*pi*(i-1)/M)*sin(2*pi*(i-1)/M)+sin(2*pi*(j-1)/N)*sin(2*pi*(j-1)/N)));  
      end  
   end
end



for i=1:M,  
    for j = 1:N,  
        d(i,j)=1/(1+(4*nu*dt/(dx*dx))*(sin(pi*(i-1)/M)*sin(pi*(i-1)/M)+sin(pi*(j-1)/N)*sin(pi*(j-1)/N)));  
    end
end
%------------------------------------------------------------------------------

for i=1:Q,
    b1(i,1)=width/2;
    b2(i,1)=(width/4) + (i-1)*ds;
end
for i=1:Q,
    bt1(i,1)=b1(i,1);
    bt2(i,1)=b2(i,1);
end

save_steps = [1 2 3 5 10 25 50 100];
save_steps = save_steps(save_steps <= nsteps);
S = struct();
S.params = struct('M', M, 'N', N, 'Q', Q, 'dx', dx, 'ds', ds, 'dt', dt, 'nu', nu, ...
    'mu', muu, 'rho', p, 'length', dx*N, 'width', width, 'time_final', time_final, ...
    'stiff', stiff, 'nsteps', nsteps, 'save_steps', save_steps);
S.X0 = [b1 b2];
S.a = a; S.b = b; S.c = c; S.d = d;

for n = 1:nsteps
    f1 = zeros(M,N);
    f2 = zeros(M,N);
    [bt1, bt2] = moveplate(bt1, bt2, Q, width);
    [fb1, fb2] = target_force(b1, b2, bt1, bt2, fb1, fb2, stiff, Q);
    forcespread(b1, b2, fb1, fb2, Q);
    fluid_solve;
    [b1, b2] = move_boundary(b1, b2, Q, width);
    time = time+dt;
    if any(save_steps == n)
        k = sprintf('step_%04d', n);
        S.(k) = struct('n', n, 'time', time, 'u', u, 'v', v, 'press', press, ...
            'f1', f1, 'f2', f2, 'w1', w1, 'w2', w2, 'X', [b1 b2], 'XT', [bt1 bt2], 'fb', [fb1 fb2]);
    end
end

save(outfile, '-struct', 'S', '-v7');
fprintf('saved %s after %d steps (time = %.10g)\n', outfile, nsteps, time);
end
