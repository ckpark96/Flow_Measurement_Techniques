clc; clear; close all; warning off;

%% WIDIM - Single pair correlation with window deformation

%% data
% reading folder and files 
foldread='C:\Users\abett\Desktop\FMT\Assignment\PIV_data\Alpha15_dt100\';        % (*** fill in ***) folder containing the images
first=1;            % (*** fill in ***) Integer number corresponding to the first image to be read 
last=100;             % (*** fill in ***) Integer number corresponding to the last image to be read 

%% Processing parameters (*** fill in ***)
ws = 32;                            % window size in pixels (scalar integer value, e.g. 16, 32, 64...)
ovlap= 0;                          % overlap [%] between 0 and 100 (integer value)
iterNum= 3;                          % number of iterations (integer >=1) for the multi-pass processing
window_shape={'square'};              % {'square'} or {'round'} (keep the curly brackets) 

% additional parameters
dt = 100;                           % time separation in microseconds
pix_size = 4.4;                     % pixel size in microns
M = 0.05488;                         % Magnification factor
xo = 128;                           % x origin in pixels
yo = 530;                           % y origin in pixels
MaskFile = 'C:\Users\abett\Desktop\FMT\Assignment\PIV_matlab_codes\WIDIM\Mask_Alpha_15.mat';     % matlab file containing the mask at that angle of attack
PlotIntermediateResults = 'no';    % 'yes' or 'no'

%% call the processing function
WIDIMtif_Proc 
disp('--------------------------------------')
disp(' ')
