clc; clear; close all;

%% statistical analysis of the velocity field data
% folder containing the velocity fields
FoldRead=''; % (*** fill in ***)
first=1;    % (*** fill in ***) first velocity field to be analyzed
last=100;     % (*** fill in ***) last velocity field to be analyzed

% do not modify the three lines below
FileRoot = 'B0';
FileApp = '.dat';
ZeroStr = '0000';

%% processing
for i=first:last
    iStr = num2str(i);
    fprintf([iStr '\n']);
    FileRead = [FileRoot ZeroStr(1:end-length(iStr)) iStr FileApp];
    [x,y,u,v,I,J] = ReadDat_2C([FoldRead FileRead]);
    % initialization
    if i==1
        uTot = zeros(J,I,last-first+1);
        vTot = zeros(J,I,last-first+1);
    end    

    uTot(:,:,i) = u;
    vTot(:,:,i) = v;
end

% compute the mean velocity components and the fluctuations
% root-mean-square (*** fill in ***)
uMean = ;
uStd = ;
vMean = ;
vStd = ;

%% create the folder foldw (where the data are written) if it doesn't exist
%% do not modify this part
FoldWrite = [FoldRead 'Stat' num2str(first) '-' num2str(last) '\'];
if exist(FoldWrite,'dir')~=7
    mkdir(FoldWrite);
end

%% write the data files (do not modify this part) 
xw = x'; yw=y'; uMeanw=uMean'; uStdw=uStd'; vMeanw=vMean'; vStdw=vStd'; 
savematrix=[xw(:) yw(:) uMeanw(:) vMeanw(:)];
savematrix(isnan(savematrix)) = 0;
fid=fopen([FoldWrite 'Mean' FileApp],'w');
fprintf(fid,'%s\n',['TITLE = "Mean"']);
fprintf(fid,'%s\n','VARIABLES = "X [mm]", "Y [mm]", "Vx [m/s]", "Vy [m/s]"');
fprintf(fid,'%s\n',['ZONE T="Frame 1", I=' num2str(I) ', J=' num2str(J) ', F=POINT']);
fprintf(fid,'%.3f %.3f %.5f %.5f\n',savematrix');
fclose(fid);

% Std
savematrix=[xw(:) yw(:) uStdw(:) vStdw(:)];
savematrix(isnan(savematrix)) = 0;
fid=fopen([FoldWrite 'Std' FileApp],'w');
fprintf(fid,'%s\n',['TITLE = "Std"']);
fprintf(fid,'%s\n','VARIABLES = "X [mm]", "Y [mm]", "Vx [m/s]", "Vy [m/s]"');
fprintf(fid,'%s\n',['ZONE T="Frame 1", I=' num2str(I) ', J=' num2str(J) ', F=POINT']);
fprintf(fid,'%.3f %.3f %.5f %.5f\n',savematrix');
fclose(fid);

%% figures 
figure(1), clf
subplot(1,2,1), contourf(x,y,uMean), axis equal, axis tight
colorbar;
xlabel('X [mm]','FontSize',14)
ylabel('Y [mm]','FontSize',14)
title(['u [m/s], Mean'],'FontSize',14)
set(gca,'FontSize',12,'Ydir','normal');
hold on
quiver(x,y,uMean,vMean,'k');

subplot(1,2,2), contourf(x,y,sqrt(uStd.^2+vStd.^2)), axis equal, axis tight
colorbar;
xlabel('X [mm]','FontSize',14)
ylabel('Y [mm]','FontSize',14)
title(['V'' RMS [m/s], Mean'],'FontSize',14)
set(gca,'FontSize',12,'Ydir','normal');