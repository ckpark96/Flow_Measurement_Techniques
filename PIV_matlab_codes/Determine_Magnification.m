clc; clear; close all;

%% data
% Folder where the calibration image is stored
FoldRead = ' '; %(*** fill in ***)
FileRead = 'B00001.tif';
pix_size =  ; % [microns] (*** fill in ***)

%% Determination of the magnification
% read the image
im = imread([FoldRead FileRead]);

% show the image using the imagesc function
figure,clf, imagesc(im), axis equal, axis tight
ttl = title('Select two reference points for calibration');
% click on two points at known distance
[x,y] =  ginput(2);
% distance between two points in pixels (*** fill in ***)
dist_px = ;
% input the distance in mm
dist = input('Distance in mm? '); 
% determine the magnification factor % (*** fill in ***)
M = ; 
fprintf(['Magnification factor = ' num2str(M,'%.5f') '\n']);

%% origin selection (in pixels)
ttl.String = 'Select the location of your origin';
[xo,yo] = ginput(1);
xo = round(xo);
yo = round(yo);
fprintf(['X origin (pixels) = ' num2str(xo) '\n']);
fprintf(['Y origin (pixels) = ' num2str(yo) '\n']);


