clc; clear; close all;
scrsz = get(0,'ScreenSize'); %Get the current screensize

%Initialize flags
ZoomFlag=0;
CorrmapFlag=0;

%Draw main window
figure('Position',[scrsz(4)/10 scrsz(4)/10 600 500],'NumberTitle','of','Name','PIV_Tool_3.0','MenuBar','None','Tag','PIVT','Resize','off');
colormap(gray(256));
HandlePIVT_Main=findobj('Tag','PIVT');

%File menu
HandlePIVT_File=uimenu('Label','&File'); %Make a parent 'File' menu  
HandlePIVT_Load=uimenu('Parent',HandlePIVT_File,'Label','Open','Accelerator','o',...
   'Callback','ioaction=''load''; PIVT_Load'); %Open a file
HandlePIVT_Cancel=uimenu('Parent',HandlePIVT_File,'Label','Exit','Accelerator','q',...
   'Callback','exit'); %Close PIVT and exit matlab

%View menu
HandlePIVT_View=uimenu('Label','&View');
HandlePIVT_Zoom=uimenu('Parent',HandlePIVT_View,'Label','Zoom','Accelerator','z',...
   'Callback','if ZoomFlag==0, zoom on, set(HandlePIVT_Zoom,''Checked'',''on''), set(gcf,''Pointer'',''crosshair'' ), ZoomFlag=1; else, zoom off, set(HandlePIVT_Zoom,''Checked'',''off''), set(gcf,''Pointer'',''arrow''), ZoomFlag=0; end ;');
HandlePIVT_Corrmap=uimenu('Parent',HandlePIVT_View,'Label','corrmap');
HandlePIVT_Corrmap_image=uimenu('Parent',HandlePIVT_Corrmap,'Label','image','Checked','on',...
    'Callback','CorrmapFlag=0; set(HandlePIVT_Corrmap_image,''Checked'',''on'');  set(HandlePIVT_Corrmap_surf,''Checked'',''off'');');
HandlePIVT_Corrmap_surf=uimenu('Parent',HandlePIVT_Corrmap,'Label','surf','Checked','off',...
    'Callback','CorrmapFlag=1; set(HandlePIVT_Corrmap_surf,''Checked'',''on'');  set(HandlePIVT_Corrmap_image,''Checked'',''off'');');


%Pointer values display
uicontrol('Style','text','String','X :','Position',[10 470 30 20]);
HandlePIVT_CoordPointX = uicontrol('Style', 'edit', 'String', '',...
  	'Position', [50 470 30 20],'Enable','inactive');	
uicontrol('Style','text','String','Y :','Position',[90 470 30 20]);
HandlePIVT_CoordPointY = uicontrol('Style', 'edit', 'String', '',...
 	'Position', [130 470 30 20],'Enable','inactive');	
uicontrol('Style','text','String','Displ :','Position',[170 470 40 20]);
HandlePIVT_DisplValueX = uicontrol('Style', 'edit', 'String', '',...
  	'Position', [220 470 50 20],'Enable','inactive');
HandlePIVT_DisplValueY = uicontrol('Style', 'edit', 'String', '',...
  	'Position', [280 470 50 20],'Enable','inactive');

%Correlation settings
uicontrol('Style','text','String','WSX :','Position',[340 470 40 20]);
HandlePIVT_WSX = uicontrol('Style', 'edit', 'String', '31',...
  	'Position', [400 470 50 20],'Enable','on');	
uicontrol('Style','text','String','WSY :','Position',[460 470 40 20]);
HandlePIVT_WSY = uicontrol('Style', 'edit', 'String', '31',...
 	'Position', [520 470 50 20],'Enable','on');	
