%Set up window for correlation map

if exist('HandlePIVT_CorrMap')>0;
    figure(HandlePIVT_CorrMap),
    set(HandlePIVT_CorrMap,'NumberTitle','of');
    set(HandlePIVT_CorrMap,'Name','Correlation map');
    set(HandlePIVT_CorrMap,'Position',[scrsz(3)/2 scrsz(4)/3 600 500],'Resize','off');
    set(HandlePIVT_CorrMap,'MenuBar','None');
else
    HandlePIVT_CorrMap=figure('Position',[scrsz(3)/2 scrsz(4)/3 600 500],'Resize','off');
    set(HandlePIVT_CorrMap,'NumberTitle','of');
    set(HandlePIVT_CorrMap,'Name','Correlation map');
    set(HandlePIVT_CorrMap,'MenuBar','None');
end;

figure(HandlePIVT_Main), imagesc(ima_b), axis xy; axis equal; axis tight
set(HandlePIVT_Main,'WindowButtonMotionFcn','PIVT_MouseMove20');
set(HandlePIVT_Main,'WindowButtonDownFcn','PIVT_MousePush'); 
drawnow;
colorbar;




