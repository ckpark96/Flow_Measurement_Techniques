/*	sincinterp2 is a function to perform sinc interpolation on an image 	*
        by: F.Schrijer and F.Scarano, to obtain the mex file type 'mex sincinterp2.c'*/

#include "mex.h"
#include "math.h"

void sincinterp2(double *image, double *dX, double *dY, double *imaold, double *pm, int rad, int dimy, int dimx, double *dima)
{
    int cy, cx;
    int xIndex, k, l;
    const double pi=3.1415927;
    double distx,disty;
    double sincx,sincy;
    double xtransf,ytransf;
    double buf;
    double xnear,ynear;
    double fractx,fracty;
    double bufImg;
    double distxsqr,dist,sinc;
    int coordx,coordy;

    
    
    for (cx=0; cx<dimx; cx++){
		xIndex=(dimy)*cx;  //For array index in dX and dY
               
		for (cy=0; cy<dimy; cy++){
			if (pm[cy+xIndex]==0){
				xtransf = dX[cy+xIndex];
				ytransf = dY[cy+xIndex];
    
				xnear = floor(xtransf);
				ynear = floor(ytransf);
				fractx = xtransf-xnear;
				fracty = ytransf-ynear;
				buf=0;

                                for (k=-rad; k<=rad; k++){
					distx=(fractx-k)*pi;
					coordx=(int) xnear+k;
					distxsqr=distx*distx;
					if (distx==0){
						sincx=1;
					}
					else{
						sincx=sin(distx)/(distx);
					}
                                        
                                        for (l=-rad; l<=rad; l++){
						disty=(fracty-l)*pi;
						if (disty==0){
							sincy=1;
						}
						else{
							sincy=sin(disty)/(disty);
						}
						coordy=(int) ynear+l;
						if ( coordx<1 || coordx>=dimx || coordy<1 || coordy>=dimy){
							bufImg=0;
						}
						else{
							bufImg=image[coordy-1+dimy*(coordx-1)];		
						}
						buf=buf+sincx*sincy*bufImg;
					}
				}
  				dima[cy+xIndex]=buf;
                                
			} // end if pm==0
		else {
			dima[cy+xIndex]=imaold[cy+xIndex];
            }  // end else
		} // for cy
   }  // for cx
}  // end of main

/* Gateway routine */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *image, *dX, *dY, *imaold, *dima, *pm;
    int rad, status, image_mrows, image_ncols, imaold_mrows, imaold_ncols, dX_mrows, dX_ncols, dY_mrows, dY_ncols, pm_mrows, pm_ncols;
    
    /* Check if the number of inputs and outputs are correct*/
    if (nrhs != 6) mexErrMsgTxt("Six inputs required.");
    if (nlhs != 1) mexErrMsgTxt("Only one output required.");
    
    /*Get the image data*/
    image = mxGetPr(prhs[0]); //Create the pointer to image
    image_mrows = mxGetM(prhs[0]); //Get the number of rows image
    image_ncols = mxGetN(prhs[0]); //Get the number of cols image
    
    /*Get the dX data*/
    dX = mxGetPr(prhs[1]); //Create the pointer to dX
    dX_mrows = mxGetM(prhs[1]); //Get the number of rows of dX
    dX_ncols = mxGetN(prhs[1]); //Get the number of cols of dX
    
    /*Get the dY data*/
    dY = mxGetPr(prhs[2]); //Create the pointer to dY
    dY_mrows = mxGetM(prhs[2]); //Get the number of rows of dY
    dY_ncols = mxGetN(prhs[2]); //Get the number of cols of dY
    
    /*Get old image data*/
    imaold = mxGetPr(prhs[3]); //Create the pointer to image
    imaold_mrows = mxGetM(prhs[3]); //Get the number of rows image
    imaold_ncols = mxGetN(prhs[3]); //Get the number of cols image
    
	 /*Get the procmaskcomb2 data*/
    pm = mxGetPr(prhs[4]); //Create the pointer to procmaskcomb2
    pm_mrows = mxGetM(prhs[4]); //Get the number of rows of procmaskcomb2
    pm_ncols = mxGetN(prhs[4]); //Get the number of cols of procmaskcomb2
    
    /*Get the interpolation radius number*/
    rad = mxGetScalar(prhs[5]); //Get the value of iter
    
    /*Set the pointer to the output image*/
    plhs[0]= mxCreateDoubleMatrix(image_mrows, image_ncols, mxREAL);
    
    /*Create a pointer to the output image*/
    dima = mxGetPr(plhs[0]);
    
    /*Call the c subroutine*/
    sincinterp2(image, dX, dY, imaold, pm, rad, image_mrows, image_ncols, dima);
      
    
}