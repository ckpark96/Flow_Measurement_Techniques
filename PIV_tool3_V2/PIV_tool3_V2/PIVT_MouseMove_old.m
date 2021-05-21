[x0,y0]=fpg(HandlePIVT_Main);
set(HandlePIVT_CoordPointY,'String',num2str(y0));
set(HandlePIVT_CoordPointX,'String',num2str(x0));
%set(HandlePIVT_CoordValue,'String',num2str(ima_b(y0,x0)));

wsx=eval(get(HandlePIVT_WSX,'String'));
wsy=eval(get(HandlePIVT_WSY,'String'));
radx=round((wsx-1)/2);
rady=round((wsy-1)/2);

if(isempty(x0))
    x0=0;
end;

if(isempty(y0))
    y0=0;
end;

if (y0>rady & y0<(dimy-rady) & x0>radx & x0<(dimx-radx))
    wa=ima(y0-rady:y0+rady,x0-radx:x0+radx);
    wb=imb(y0-rady:y0+rady,x0-radx:x0+radx);
    
    a2=wa-mean(wa(:));
    b2=wb-mean(wb(:));
    Nfftx=2^( ceil(log2(wsx+radx)) );
    Nffty=2^( ceil(log2(wsy+rady)) );
    b2 = b2(end:-1:1,end:-1:1); 
    ffta=fft2(a2,Nffty,Nfftx);
    fftb=fft2(b2,Nffty,Nfftx);
    ffta(1:2,Nfftx/2-1:Nfftx/2+2)=0;
    ffta(end-1:end,Nfftx/2-1:Nfftx/2+2)=0;
    fftb(1:2,Nfftx/2-1:Nfftx/2+2)=0;
    fftb(end-1:end,Nfftx/2-1:Nfftx/2+2)=0;
    c = real(ifft2(ffta.*fftb))/(wsx*wsy);
    corrmap=c(wsy-rady+1:wsy+rady,wsx-radx+1:wsx+radx);
    clear bufw;
    bufw(:,:,1)=flipud(wa);
    bufw(:,:,2)=flipud(wb);
    bufw(:,:,3)=flipud(wb);
    bufw=bufw/(max(bufw(:)));
    
    
    corrmap(find(isnan(corrmap)))=0;
    [peaks,sn]=analyse_map_tool(corrmap,1,3,1);
    sn=max(sn,1);
    
    
    figure(HandlePIVT_CorrMap), subplot(1,2,1), imagesc(bufw), axis equal, title(['wa+wb']), drawnow;
    if CorrmapFlag==0
        figure(HandlePIVT_CorrMap), subplot(1,2,2), imagesc(flipud(corrmap)), axis equal, title(['Cmap S/N=' num2str(0.1*round(sn*10))]), drawnow;
    end;
    if CorrmapFlag==1
        figure(HandlePIVT_CorrMap), subplot(1,2,2), surf(flipud(corrmap)), title(['Cmap S/N=' num2str(0.1*round(sn*10))]), drawnow;
    end;
end;