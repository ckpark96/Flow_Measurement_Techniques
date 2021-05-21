path(path,pwd); %Set program location in the Matlab-path to find subprogz
[filename, pathname] = uigetfile('*.tif', 'Open an image-file'); 
path(path, pathname); 
Path2=['cd('''  pathname ''')'];
%eval(['cd ' pathname]); %Sets current dir as the place where the last loaded file was
eval(Path2);
drawnow; 

filename_ima=filename;
filename_imb=filename;

filename_ima(find(filename=='.')-1)='a';
filename_ima(find(filename=='.')-1)='b';

ima=flipud(double(imread(filename_ima)));
imb=flipud(double(imread(filename_imb)));
[dimy,dimx]=size(ima);
ima_b=(ima+imb)/2;

PIVT_Window; %Draw image and set up correlation window

