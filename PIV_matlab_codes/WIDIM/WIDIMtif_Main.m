clc; clear; close all; warning off;

%% WIDIM - Single pair correlation with window deformation

%% data
% reading folder and files 
foldread='';        % (*** fill in ***) folder containing the images
first=;            % (*** fill in ***) Integer number corresponding to the first image to be read 
last=;             % (*** fill in ***) Integer number corresponding to the last image to be read 

%% Processing parameters (*** fill in ***)
ws = ;                            % window size in pixels (scalar integer value, e.g. 16, 32, 64...)
ovlap= ;                          % overlap [%] between 0 and 100 (integer value)
iterNum=;                          % number of iterations (integer >=1) for the multi-pass processing
window_shape={''};              % {'square'} or {'round'} (keep the curly brackets) 

% additional parameters
dt = ;                           % time separation in microseconds
pix_size = ;                     % pixel size in microns
M = ;                         % Magnification factor
xo = ;                           % x origin in pixels
yo = ;                           % y origin in pixels
MaskFile = '';     % matlab file containing the mask at that angle of attack
PlotIntermediateResults = '';    % 'yes' or 'no'

%% call the processing function
WIDIMtif_Proc 
disp('--------------------------------------')
disp(' ')
