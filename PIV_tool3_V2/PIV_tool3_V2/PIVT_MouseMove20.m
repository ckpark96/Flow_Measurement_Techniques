x0d=1;
while x0d>0
    [x0,y0]=fpg(HandlePIVT_Main);
    set(HandlePIVT_CoordPointY,'String',num2str(y0));
    set(HandlePIVT_CoordPointX,'String',num2str(x0));
    %set(HandlePIVT_CoordValue,'String',num2str(ima_b(y0,x0)));
    
    wsx=eval(get(HandlePIVT_WSX,'String'));
    wsy=eval(get(HandlePIVT_WSY,'String'));
    if mod(wsx,2)==0 wsx=wsx+1; end
    if mod(wsy,2)==0 wsy=wsy+1; end
    radx=round((wsx-1)/2);
    rady=round((wsy-1)/2);
    
    if(isempty(x0))
        x0=0;
    end;
    
    if(isempty(y0))
        y0=0;
    end;
    
    
    
    d=5;
    r=(d-1)/2;
    if (y0>(r+1)*rady && y0<(dimy-(r+1)*rady) && x0>(r+1)*radx && x0<(dimx-(r+1)*radx))
        
        
        % standard cross correlation
        [bufw1,displx1,disply1,corrmap1,sn1]=std_analysis(x0,y0,radx,rady,ima,imb);
        
        % 1-step deformed cross correlation
        [bufw2,displx2,disply2,corrmap2,sn2]=def1_analysis(x0,y0,radx,rady,ima,imb);
        
        % multi-step deformed cross correlation
        %[bufw3,displx3,disply3,corrmap3,sn3]=def2_analysis(x0,y0,radx,rady,ima,imb);
        
        figure(HandlePIVT_CorrMap),clf, subplot(2,2,1), imagesc(bufw1), axis equal, axis tight, title(['wa+wb']);% drawnow;
        figure(HandlePIVT_CorrMap), subplot(2,2,3), imagesc(bufw2), axis equal, axis tight, title(['wa+wb def']);% drawnow;
        if CorrmapFlag==0
            figure(HandlePIVT_CorrMap), subplot(2,2,2), imagesc(flipud(corrmap1)), axis equal, axis tight, title(['Cmap S/N=' num2str(0.1*round(sn1*10))]);% drawnow;
            figure(HandlePIVT_CorrMap), subplot(2,2,4), imagesc(flipud(corrmap2)), axis equal, axis tight, title(['Cmap S/N def=' num2str(0.1*round(sn2*10))]);% drawnow;
        elseif CorrmapFlag==1
            figure(HandlePIVT_CorrMap), subplot(2,2,2), surf(flipud(corrmap1)), title(['Cmap S/N=' num2str(0.1*round(sn1*10))]);% drawnow;
            figure(HandlePIVT_CorrMap), subplot(2,2,4), surf(flipud(corrmap2)), title(['Cmap S/N def=' num2str(0.1*round(sn2*10))]);% drawnow;
        end;
        
    end; % if
    % compute and print the displacement
    PIVT_MousePush
%     [x_mp,y_mp]=PIVT_MousePush(HandlePIVT_Main);
end % while

