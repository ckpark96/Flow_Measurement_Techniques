clc; clear; close all;

%% data
% Folder where the calibration image is stored
FoldRead = 'C:\Users\abett\Desktop\FMT\Assignment\PIV_data\Calibration\'; %(*** fill in ***)
FileRead = 'B00001.tif';
pix_size =  4.4; % [microns] (*** fill in ***)

%% Determination of the magnification
% read the image
im = imread([FoldRead FileRead]);

% show the image using the imagesc function
figure,clf, imagesc(im), axis equal, axis tight
ttl = title('Select two reference points for calibration');
% click on two points at known distance
[x,y] =  ginput(2);
% distance between two points in pixels (*** fill in ***)
dist_px = ( (x(2)-x(1))^2 + (y(2)-y(1))^2 )^0.5 ;
% input the distance in mm
dist = input('Distance in mm? '); 
% determine the magnification factor % (*** fill in ***)
M = ( dist_px*pix_size*10^(-6) ) / ( dist*10^(-3) ); 
fprintf(['Magnification factor = ' num2str(M,'%.5f') '\n']);

%% origin selection (in pixels)
ttl.String = 'Select the location of your origin';
[xo,yo] = ginput(1);
xo = round(xo);
yo = round(yo);
fprintf(['X origin (pixels) = ' num2str(xo) '\n']);
fprintf(['Y origin (pixels) = ' num2str(yo) '\n']);


