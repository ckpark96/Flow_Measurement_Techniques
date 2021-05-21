function [x,y,u,v,lenx,leny] = ReadDat_2C(FileRead)

fid=fopen(FileRead); %read the file
for i=1:3 tline = fgetl(fid); end
equalsign = findstr(tline, '=');
commasign = findstr(tline, ',');
lenx=str2num(tline(equalsign(2)+1:commasign(2)-1));
leny=str2num(tline(equalsign(3)+1:commasign(3)-1));
M = dlmread(FileRead,' ', 3, 0);
fclose(fid);


x=reshape(M(:,1),lenx,leny)';
y=reshape(M(:,2),lenx,leny)';
u=reshape(M(:,3),lenx,leny)';
v=reshape(M(:,4),lenx,leny)';

end % end function