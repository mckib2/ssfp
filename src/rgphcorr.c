
/* ............................................................ */
/*	Region-Growing Phase Correction for Balanced SSFP	*/
/*								*/
/*	Brian A. Hargreaves, May 2004				*/
/* ............................................................ */

/* ------------------------------------------------------------
        $log$
   ------------------------------------------------------------ */



#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef PI
	#define PI 3.14159
#endif
/* #define DEBUG */
/* #define COUNTUSE */
/* #define DEBUGFINE */




void celloffset2xyz(cimagetype *cim, long celloffset, 
			short *cx, short *cy, short *cz)
/* ............................................................ */
/*	Convert a linear cell offset (0-cim->ncells) to 	*/
/*	(cx,cy,cz) coordinates that are numbered from		*/
/*	(1,1,1) to (cim->nxcells,cim->nycells,cim->nzcells).	*/
/*								*/
/*	Thus, offset = (cz-1)*ny*nx + (cy-1)*nx + (cx-1)	*/
/*								*/
/*		where nx = cim->nxcells, etc.			*/
/*								*/
/*								*/
/*	INPUT:							*/
/*		cim		- complex image structure	*/
/*		celloffset	- linear offset to cell		*/
/*								*/
/*	OUTPUT:							*/
/*		cx,cy,cz	- cell coordinates		*/
/* ............................................................ */

{
*cx = celloffset % cim->nxcells + 1;
*cy = (celloffset/cim->nxcells) % cim->nycells+1;
*cz = (celloffset/cim->nxcells/cim->nycells) % cim->nzcells+1;
}



int partition( long *order, float *numlist, long bot, long top)
/* ............................................................ */
/*	Partition function for quick sort.  (Fairly standard)	*/
/*	Takes the first element of an array as the partition	*/
/*	point, and inserts it into the array in a position so	*/
/*	that all of the points below it have values less than   */
/*	it, and vice versa.					*/
/*								*/
/*	INPUT:							*/
/*		order	- original indices of array, get 	*/
/*			  moved to correspond with values 	*/
/*		numlist - list to sort.				*/
/*		bot	- bottom index of numlist/order		*/
/*		top	- top index of numlist/order		*/
/*								*/
/*	OUTPUT:							*/
/*		(value) - partition index			*/
/*		order	- original indices of array, get 	*/
/*			  moved to correspond with values 	*/
/*		numlist - list to sort.				*/
/* ............................................................ */
   	{
   	float pivot, t;
	long i, j, q;
   	pivot = numlist[bot];
	i = bot; j = top+1;
		
	while( 1)
		{
		do ++i; while( numlist[i] <= pivot && i < top );
		do --j; while( numlist[j] > pivot );
		if( i >= j ) break;
		t = numlist[i]; numlist[i] = numlist[j]; numlist[j] = t;
		q = order[i]; order[i] = order[j]; order[j] = q;
		}
	t = numlist[bot]; numlist[bot] = numlist[j]; numlist[j] = t;
	q = order[bot]; order[bot] = order[j]; order[j] = q;

   	return j;
	}



void quicksort( long *order, float *numlist, long bot, long top)
/* ............................................................ */
/*	Quick-Sort function, with indices.			*/
/*	Sorts an array (numlist) using the quick-sort algorithm */
/*	and also manipulates an array if indices (order) at 	*/
/*	the same time.						*/
/*								*/
/*	INPUT:							*/
/*		order	- original indices of array, get 	*/
/*			  moved to correspond with values 	*/
/*		numlist - list to sort.				*/
/*		bot	- bottom index of numlist/order		*/
/*		top	- top index of numlist/order		*/
/*								*/
/*	OUTPUT:							*/
/*		order	- original indices of array, get 	*/
/*			  moved to correspond with values 	*/
/*		numlist - sorted list.				*/
/*								*/
/*	This is a recursive function, and uses the function	*/
/*	partition(....)						*/
/* ............................................................ */
{
long part;

if( bot < top ) 
	{
      	part = partition( order, numlist, bot, top);
   	quicksort( order, numlist, bot, part-1);
   	quicksort( order, numlist, part+1, top);
	}
}






void print_stepinfotype( cimagetype *cim)
/* ............................................................ */
/*	Function prints values of a stepinfotype structure.     */
/* ............................................................ */
	{
	printf("Step Info:\n");
	printf("  Increments within cell = (%3d,%3d,%3d)  \n",
			cim->stepinfo->xstep, cim->stepinfo->ystep, 
			cim->stepinfo->zstep );
	printf("  Offset Increments move to new cell  = (%3d,%3d,%3d)  --  %d total\n",
			cim->stepinfo->cxstep, cim->stepinfo->cystep, 
			cim->stepinfo->czstep );
		
	}


	
void print_cimagetype( cimagetype *cim)
/* ............................................................ */
/*	Function prints values of a cimagetype structure.       */
/* ............................................................ */
	{
	printf("Complex Image Type Info:\n");
	printf("  Data Size = (%3d,%3d,%3d)  --  %d total\n",
			cim->xsize,cim->ysize,cim->zsize,cim->npts);
	printf("  Cell Size = (%3d,%3d,%3d) \n",cim->cxsize,
					cim->cysize,cim->czsize);
	printf("  Num Cells = (%3d,%3d,%3d)  --  %d total\n",cim->nxcells,
					cim->nycells,cim->nzcells,cim->ncells);
	}



int init_cellinfo( cimagetype *cimage )
/* ............................................................ */
/*	Initialize the information for stepping through image   */
/*	arrays and cell array.  				*/
/*								*/
/*	INPUT:							*/
/*								*/
/*	OUTPUT:							*/
/* ............................................................ */

{
long cxcount;
long cycount;
long czcount;
short cx, cy, cz;
short zborder, yzborder;
cellinfotype *cellptr;

cellptr = cimage->cellinfo;

for (czcount=0; czcount < cimage->nzcells; czcount++)
	{
	zborder = ((czcount==0) || (czcount==cimage->nzcells-1));
	for (cycount=0; cycount < cimage->nycells; cycount++)
		{
		yzborder = (zborder || (cycount==0) || (cycount==cimage->nycells-1));

		cellptr->done = 0.0;
		cellptr->border = 1;	
		cellptr++;
		for (cxcount=1; cxcount < cimage->nxcells-1; cxcount++)
			{
			cellptr->done = 0.0;
			cellptr->border = yzborder;	
			cellptr++;
			}
		cellptr->done = 0.0;
		cellptr->border = 1;	
		cellptr++;
		}	
	}

#ifdef COUNTUSE
	cellptr = cimage->cellinfo;

	for (czcount=0; czcount < cimage->ncells; czcount++)
		(cellptr++)->numused = 0;
#endif

}


	
int init_stepinfo( cimagetype *cimage)
/* ............................................................ */
/*	Initialize the information for stepping through image   */
/*	arrays and cell array.  				*/
/*								*/
/*	INPUT:							*/
/*								*/
/*	OUTPUT:							*/
/* ............................................................ */
{
	/* Steps - amount to increment pointers at the end of x, y, z
		loops to move through a cell.	*/

cimage->stepinfo->xstep = 1;	/* Simple! */
cimage->stepinfo->ystep = cimage->xsize - cimage->cxsize;	
cimage->stepinfo->zstep = cimage->xsize * (cimage->ysize - cimage->cysize);

	/* Cell steps - amount to step in data array after finishing 
		a cell to increment the cell in x, y or z. 	
		For y, x will have happened.  
		For z, x and y will have happened.  */

cimage->stepinfo->cxstep = cimage->cxsize -
			cimage->czsize * (cimage->xsize * cimage->ysize);
cimage->stepinfo->cystep = (cimage->cysize -1) * cimage->xsize;
cimage->stepinfo->czstep = (cimage->czsize -1) * cimage->xsize * cimage->ysize;

cimage->stepinfo->celldx = 1;			/* Offset to next cell in x. */
cimage->stepinfo->celldy = cimage->nxcells; 	/* Offset to next cell in y. */
cimage->stepinfo->celldz = cimage->nxcells * cimage->nycells;	
						/* Offset to next z cell.*/
}






void scatterfitcell(cimagetype *cim, 
			float **rptr, float **iptr, float *scatterangle,
			float *avgmagnitude)
/* ............................................................ */
/*	Traverse a cell, finding the angle of the line that     */
/*	passes through 0,0 and minimizes the sum of squares of  */
/*	orthogonal distances to scatter points.	 There is a	*/
/*	deliberate 180-degree ambiguity in this angle.		*/
/*								*/
/*	INPUT:							*/
/*		cim	- complex image structure		*/
/*		rptr	- pointer to real data offset (in cim)  */
/*		iptr	- pointer to imag data offset (in cim)  */
/*								*/
/*	OUTPUT:							*/
/*		scatterangle - fitted scatter angle in rad	*/
/*		avgmagnitude - average cell magnitude		*/
/*								*/
/*	NOTE:							*/
/*		It *IS* important to use atan2, ie the 2D	*/
/*		arctan function.  atan(y/x) instead can 	*/
/*		erroneously give the line that MAXIMIZES	*/
/*		the sum of squared distances...			*/
/*								*/
/* ............................................................ */

{
int xcount;
int ycount;
int zcount;

double sxx=0;
double syy=0;
double sxy=0;

double smag=0;


for (zcount = 0; zcount < cim->czsize; zcount++)
	{
	for (ycount = 0; ycount < cim->cysize; ycount++)
		{
		for (xcount = 0; xcount < cim->cxsize; xcount++)
			{

			sxx += (**rptr) * (**rptr);
			syy += (**iptr) * (**iptr);
			sxy += (**rptr) * (**iptr);
			smag += sqrt( (**rptr) * (**rptr) + (**iptr) * (**iptr) );

			(*rptr) += cim->stepinfo->xstep;
			(*iptr) += cim->stepinfo->xstep;
			}
		(*rptr) += cim->stepinfo->ystep;
		(*iptr) += cim->stepinfo->ystep;
		}
	(*rptr) += cim->stepinfo->zstep;
	(*iptr) += cim->stepinfo->zstep;
	}			

*scatterangle = (float) (0.5 * atan2(2*sxy, sxx-syy) );
*avgmagnitude = (float) (smag / cim->ncpts );

}


void dofullscatterfit( cimagetype *cim )
/* ............................................................ */
/*	Traverse all cells, finding the angle of the line that  */
/*	passes through 0,0 and minimizes the sum of squares of  */
/*	orthogonal distances to scatter points.	 There is a	*/
/*	deliberate 180-degree ambiguity in this angle.		*/
/*	The fitted scatter angle and magnitude are stored in	*/
/*	the cell info array, and the done field is set to 0.    */
/*								*/
/*	INPUT:							*/
/*		cim	- complex image structure		*/
/*								*/
/*	OUTPUT:							*/
/*		cim->cellinfo[] is updated with information     */
/*			from the fit.				*/
/* ............................................................ */
{
int xccount;
int yccount;
int zccount;

long ccount=0;

float *rdat;
float *idat;
cellinfotype *cellptr;
float scatterang;
float avgmag;

rdat = cim->rdata;
idat = cim->idata;
cellptr = cim->cellinfo;



for (zccount = 0; zccount < cim->nzcells; zccount++)
	{
	for (yccount = 0; yccount < cim->nycells; yccount++)
		{
		for (xccount = 0; xccount < cim->nxcells; xccount++)
			{
			scatterfitcell(cim, &rdat, &idat, 
						&scatterang, &avgmag);
			
			cellptr->scatterphase = scatterang;
			cellptr->mag = avgmag;
			cellptr->done = 0;

			cellptr++;
			rdat += cim->stepinfo->cxstep;
			idat += cim->stepinfo->cxstep;

			ccount++;
			}
		rdat += cim->stepinfo->cystep;
		idat += cim->stepinfo->cystep;
		}
	rdat += cim->stepinfo->czstep;
	idat += cim->stepinfo->czstep;
	}			
#ifdef DEBUG
  printf("scatter fit done to %d cells.\n",ccount);
#endif
}



void rotatecell (cimagetype *cim, float **rptr, float **iptr, 
			float rotcos, float rotsin)
/* ............................................................ */
/*	Traverse a cell, rotating all data points in the 	*/
/*	complex plane by the angle expressed by rotcos and	*/
/*	rotsin.							*/
/*								*/
/*	INPUT:							*/
/*		cim	- complex image structure		*/
/*		rptr	- real data pointer.			*/
/*		iptr	- imag data pointer.			*/
/*		rotcos - cosine of rotation angle.		*/
/*		rotsin - sine of rotation angle.		*/
/*								*/
/*	OUTPUT:							*/
/*		rptr - updated real data pointer		*/
/*		iptr - updated imag data pointer		*/
/*		(real/imag data points are updated		*/
/* ............................................................ */

{
int xcount;
int ycount;
int zcount;

float realdat;


for (zcount = 0; zcount < cim->czsize; zcount++)
	{
	for (ycount = 0; ycount < cim->cysize; ycount++)
		{
		for (xcount = 0; xcount < cim->cxsize; xcount++)
			{
			realdat = rotcos * (**rptr) - rotsin * (**iptr);
			(**iptr) = rotcos * (**iptr) + rotsin * (**rptr);
			(**rptr) = realdat;

			(*rptr) += cim->stepinfo->xstep;
			(*iptr) += cim->stepinfo->xstep;
			}
		(*rptr) += cim->stepinfo->ystep;
		(*iptr) += cim->stepinfo->ystep;
		}
	(*rptr) += cim->stepinfo->zstep;
	(*iptr) += cim->stepinfo->zstep;
	}			
}



void rotateallcells( cimagetype *cim )
/* ............................................................ */
/*	Traverse all cells, removing the phase angle that 	*/
/*	is in the cellinfo array from each data point in the	*/
/*	cell.							*/
/*								*/
/*	INPUT:							*/
/*		cim	- complex image structure		*/
/*								*/
/*	OUTPUT:							*/
/*		cim->rdata and idata are updated.		*/
/* ............................................................ */
{
int xccount;
int yccount;
int zccount;

float *rdat;
float *idat;
cellinfotype *cellptr;

float rotcos, rotsin;
long ccount=0;

rdat = cim->rdata;
idat = cim->idata;
cellptr = cim->cellinfo;


for (zccount = 0; zccount < cim->nzcells; zccount++)
	{
	for (yccount = 0; yccount < cim->nycells; yccount++)
		{
		for (xccount = 0; xccount < cim->nxcells; xccount++)
			{
				/* Negate rotsin, to do undo phase */
			rotcos = cos(cellptr->alignedphase);
			rotsin = -sin(cellptr->alignedphase);

			rotatecell(cim,&rdat, &idat, rotcos,rotsin);

			cellptr++;
			rdat += cim->stepinfo->cxstep;
			idat += cim->stepinfo->cxstep;
			ccount++;
			}
		rdat += cim->stepinfo->cystep;
		idat += cim->stepinfo->cystep;
		}
	rdat += cim->stepinfo->czstep;
	idat += cim->stepinfo->czstep;
	}			
}






int cimage_free( cimagetype *cim)
/* ............................................................ */
/*	Free space allocated for the complex image type.	*/
/*	size variables within the structure.			*/
/*								*/
/*	INPUT:							*/
/*		cim	- complex image structure		*/
/* ............................................................ */
{
free(cim->rdata);
free(cim->idata);
free(cim->cellinfo);
free(cim->stepinfo);
}




int cimage_allocate( cimagetype *cim, short nx, short ny, short nz,
			short cx, short cy, short cz)
/* ............................................................ */
/*	Allocate space for the complex image type, and set	*/
/*	size variables within the structure.			*/
/*								*/
/*	INPUT:							*/
/*		cim	- complex image structure		*/
/*		nx,ny,nz - sizes in x,y,z			*/
/*		cx,cy,cz - cell sizes in x,y,z			*/
/*								*/
/*	OUTPUT:							*/
/*		cim 	- structure is updated. 		*/
/* ............................................................ */

{
cim->xsize = nx;
cim->ysize = ny;
cim->zsize = nz;
cim->npts = (long)nx* (long)ny* (long)nz;
cim->rdata = (float *) malloc( cim->npts * sizeof(float));
cim->idata = (float *) malloc( cim->npts * sizeof(float));

cim->cxsize = cx;
cim->cysize = cy;
cim->czsize = cz;
cim->ncpts = (long)cx* (long)cy* (long)cz;
cim->nxcells = cim->xsize / cim->cxsize;
cim->nycells = cim->ysize / cim->cysize;
cim->nzcells = cim->zsize / cim->czsize;
cim->ncells = (long)cim->nxcells * (long)cim->nycells * (long)cim->nzcells;

cim->cellinfo = (cellinfotype *) malloc( cim->ncells * sizeof(cellinfotype));
cim->stepinfo = (stepinfotype *) malloc( sizeof(stepinfotype));


/*	Error Checks 	*/


if ((cim->nxcells * cim->cxsize) != cim->xsize)
	printf(">>> Error - x size not a multiple of cell size.\n");
if ((cim->nycells * cim->cysize) != cim->ysize)
	printf(">>> Error - y size not a multiple of cell size.\n");
if ((cim->nzcells * cim->czsize) != cim->zsize)
	printf(">>> Error - z size not a multiple of cell size.\n");


init_cellinfo( cim );
init_stepinfo( cim );

return 0;	
}

void setupneighbourignore(short cx, short cy, short cz,
			short ncx, short ncy, short ncz,
			short *neighbourignore)
/* ............................................................ */
/*	Given cell coordinates (cx,cy,cz) numbered from 1 to    */
/*	ncx, ncy or ncz, setup the neighbourignore[] array that */
/*	has 1 where the neighbour cell should be ignored because*/
/* 	the current cell is on the image border.  Other values  */
/* 	in the array are set to 0.				*/
/*								*/
/*	INPUT:							*/
/*		cx,cy,cz - Cell coordinates, 1 to ncx,ncy,ncz   */
/*		ncx,ncy,ncz - Number of cells in x,y,z in image */
/*								*/
/*	OUTPUT:							*/
/*		neighbourignore - array of neighbours to ignore */
/* ............................................................ */
{
int ncount;

for (ncount = 0; ncount < 26; ncount++)
	neighbourignore[ncount]=0;

if (cx == 1)		/* Ignore neighbours with smaller x */
	{
	neighbourignore[0] = 1;
	neighbourignore[3] = 1;
	neighbourignore[6] = 1;
	neighbourignore[9] = 1;
	neighbourignore[12]= 1;
	neighbourignore[14]= 1;
	neighbourignore[17]= 1;
	neighbourignore[20]= 1;
	neighbourignore[23]= 1;
	}
else if (cx == ncx)  	/* Ignore neighbours with bigger x*/
	{
	neighbourignore[2] = 1;
	neighbourignore[5] = 1;
	neighbourignore[8] = 1;
	neighbourignore[11]= 1;
	neighbourignore[13]= 1;
	neighbourignore[16]= 1;
	neighbourignore[19]= 1;
	neighbourignore[22]= 1;
	neighbourignore[25]= 1;
	}

if (cy == 1)		/* Ignore neighbours with smaller y */
	{
	neighbourignore[0] = 1;
	neighbourignore[1] = 1;
	neighbourignore[2] = 1;
	neighbourignore[9] = 1;
	neighbourignore[10]= 1;
	neighbourignore[11]= 1;
	neighbourignore[17]= 1;
	neighbourignore[18]= 1;
	neighbourignore[19]= 1;
	}
else if (cy == ncy)  	/* Ignore neighbours with bigger y*/
	{
	neighbourignore[6] = 1;
	neighbourignore[7] = 1;
	neighbourignore[8] = 1;
	neighbourignore[14]= 1;
	neighbourignore[15]= 1;
	neighbourignore[16]= 1;
	neighbourignore[23]= 1;
	neighbourignore[24]= 1;
	neighbourignore[25]= 1;
	}

if (cz == 1)		/* Ignore neighbours with smaller z */
	{
	for (ncount=0; ncount<9; ncount++)
		neighbourignore[ncount] = 1;
	}
else if (cz == ncz)  	/* Ignore neighbours with bigger z*/
	{
	for (ncount=17; ncount<26; ncount++)
		neighbourignore[ncount] = 1;
	}

}



#ifdef COUNTUSE
void printcelluse(cimagetype *cim)
/* ............................................................ */
/*	Print out a list of the cells, and how many times 	*/
/*	they were used as a neighbour for phase alignment.	*/
/*	This is for debugging purposes only.			*/
/* ............................................................ */
{
int xccount,yccount,zccount;
long cellnum=0;

for (zccount=0; zccount < cim->nzcells; zccount++)
	for (yccount=0; yccount < cim->nycells; yccount++)
		for (xccount=0; xccount < cim->nxcells; xccount++)
			{
			if (cim->cellinfo[cellnum].border ==1)
				printf("Border ");
			else
				printf("       ");
	
			printf("Cell (%d,%d,%d) uses = %d\n",
						xccount+1,yccount+1,zccount+1,
					cim->cellinfo[cellnum].numused);
			cellnum++;
			}
}
#endif

int rgphasealign(cimagetype *cim, long *cellorder)
/* ............................................................ */
/*	Go through the cells in the order given, aligning	*/
/*	the phase of a cell to within 90 degrees of the		*/
/*	weighted-sum-vector from the (26) nearest cells.	*/
/*	Cell weights are included only after that cell phase	*/
/*	has been aligned.					*/
/*								*/
/*	INPUT:							*/
/*		cim	- complex image structure		*/
/*		cellorder - offsets of cells to align, in order */
/*								*/
/*	OUTPUT:							*/
/*		cim 	- structure is updated. 		*/
/* ............................................................ */

{
long ccount;
long ncount;
long noffsets[26];	/* Offsets in sequential cell list to 26 neighbours */
short nignore[26];	/* Flag to ignore certain neighbours on borders. */
short *nignoreptr;	/* Pointer for quick traverse */

float rwt;
float iwt;
float dotproduct;

cellinfotype *cellinfoptr;
cellinfotype *neighbourptr;
long *noffptr;

short cellx, celly, cellz;
long cellnum;

  /* Setup offsets for neighbour cells */

noffsets[0]= -cim->stepinfo->celldx -cim->stepinfo->celldy - 
			cim->stepinfo->celldz;
noffsets[1]= cim->stepinfo->celldx;
noffsets[2]= cim->stepinfo->celldx;
noffsets[3]= -2*cim->stepinfo->celldx + cim->stepinfo->celldy;
noffsets[4]= cim->stepinfo->celldx;
noffsets[5]= cim->stepinfo->celldx;
noffsets[6]= -2*cim->stepinfo->celldx + cim->stepinfo->celldy;
noffsets[7]= cim->stepinfo->celldx;
noffsets[8]= cim->stepinfo->celldx;
noffsets[9]= -2*cim->stepinfo->celldx - 2*cim->stepinfo->celldy + 
			cim->stepinfo->celldz;
noffsets[10]= cim->stepinfo->celldx;
noffsets[11]= cim->stepinfo->celldx;
noffsets[12]= -2*cim->stepinfo->celldx + cim->stepinfo->celldy;
noffsets[13]= 2*cim->stepinfo->celldx;			/* Skip center!! */
noffsets[14]= -2*cim->stepinfo->celldx + cim->stepinfo->celldy;
noffsets[15]= cim->stepinfo->celldx;
noffsets[16]= cim->stepinfo->celldx;
noffsets[17]= -2*cim->stepinfo->celldx - 2*cim->stepinfo->celldy + 
			cim->stepinfo->celldz;
noffsets[18]= cim->stepinfo->celldx;
noffsets[19]= cim->stepinfo->celldx;
noffsets[20]= -2*cim->stepinfo->celldx + cim->stepinfo->celldy;
noffsets[21]= cim->stepinfo->celldx;
noffsets[22]= cim->stepinfo->celldx;
noffsets[23]= -2*cim->stepinfo->celldx + cim->stepinfo->celldy;
noffsets[24]= cim->stepinfo->celldx;
noffsets[25]= cim->stepinfo->celldx;

#ifdef DEBUG
	for (ncount = 0; ncount < 26; ncount++)
		printf("Neighbour offset [%d] = %d \n",ncount,noffsets[ncount]);
#endif

  /* Start with first cell */

cellinfoptr = cim->cellinfo+ *cellorder;	
cellinfoptr->realwt = cellinfoptr->mag * cos(cellinfoptr->scatterphase);
cellinfoptr->imagwt = cellinfoptr->mag * sin(cellinfoptr->scatterphase);
cellinfoptr->alignedphase = cellinfoptr->scatterphase;
cellinfoptr->done = 1.0;

#ifdef DEBUG
	celloffset2xyz(cim, *cellorder,&cellx,&celly,&cellz);
  	printf("  Starting Cell is (%3ld,%3ld,%3ld) \n",cellx,celly,cellz);
#endif

for (ccount = 1; ccount < cim->ncells; ccount++)
	{
	rwt = 0;
	iwt = 0;

	cellnum = cellorder[ccount];
	cellinfoptr = cim->cellinfo + cellorder[ccount];
	cellinfoptr->realwt = cellinfoptr->mag * cos(cellinfoptr->scatterphase);
	cellinfoptr->imagwt = cellinfoptr->mag * sin(cellinfoptr->scatterphase);
	cellinfoptr->alignedphase = cellinfoptr->scatterphase;
	neighbourptr = cellinfoptr;
	noffptr = &(noffsets[0]);

	#ifdef DEBUG
	  celloffset2xyz(cim, cellorder[ccount],&cellx,&celly,&cellz);
	  if (cellinfoptr->border==0)
		printf("    ");
	  else
		printf("    Border ");
  	  printf("Cell is (%3ld,%3ld,%3ld), offset = %ld \n",cellx,celly,cellz,
			cellnum);
	
	#endif

		/* Add up complex weights from neighbours */

	if (cellinfoptr->border==0)	/* Non-border - use all neighbours */
	    {
	    for (ncount = 0; ncount < 26; ncount++)
		{
		cellnum += *noffptr;
		neighbourptr += *noffptr++;
		rwt += neighbourptr->realwt * neighbourptr->done;
		iwt += neighbourptr->imagwt * neighbourptr->done;
		#ifdef COUNTUSE
			(neighbourptr->numused)++;
		#endif
		#ifdef DEBUGFINE
	  	  celloffset2xyz(cim, cellnum,&cellx,&celly,&cellz);
		  printf("******Neighbour Cell is (%3ld,%3ld,%3ld) \n",cellx,celly,cellz);
		#endif
		}
	    }

	else	/* Border cell... so figure out which neighbours to not use.*/
		/* 							*/
		/* Note that the main alignment (not the else) does	*/
		/* not really know where a cell or its neighbours are,  */
		/* (in x,y,z) but just uses a single offset to the 	*/
		/* neighbouring cells.  This is fast and works fine, 	*/
		/* except on borders.					*/
		/*							*/
		/* The code here does the border cells more slowly, 	*/
		/* checking which neighbours to ignore, as the offsets	*/
		/* for those will wrap across the edges of the 3D image.*/
	
	    {
	    celloffset2xyz(cim, cellorder[ccount],&cellx,&celly,&cellz);

		/* Set up nignore[] to ignore certain neighbours */

	    setupneighbourignore(cellx,celly,cellz,cim->nxcells,cim->nycells,
					cim->nzcells, nignore);
	 
	    nignoreptr = &(nignore[0]);

	    for (ncount = 0; ncount < 26; ncount++)
		{
		cellnum += *noffptr;
		neighbourptr += *noffptr++;

		if (!(*nignoreptr++))
			{
	    		#ifdef DEBUG
				printf("    	Using neighbour %d\n",ncount);
	    		#endif
			rwt += neighbourptr->realwt * neighbourptr->done;
			iwt += neighbourptr->imagwt * neighbourptr->done;
			#ifdef COUNTUSE
		  	  (neighbourptr->numused)++;
			#endif
			#ifdef DEBUGFINE
	    			celloffset2xyz(cim, cellnum,
						&cellx,&celly,&cellz);
		  		printf("******Neighbour Cell is (%3ld,%3ld,%3ld) \n",cellx,celly,cellz);
				
			#endif
			}
		}
	    }

	cellinfoptr->done = 1.0;

		/* Calculate dot product - if negative, invert cell phase. */

	dotproduct = (cellinfoptr->realwt * rwt) + (cellinfoptr->imagwt * iwt);	
	if (dotproduct < 0)
		{
		cellinfoptr->realwt *= -1.0;
		cellinfoptr->imagwt *= -1.0;
		cellinfoptr->alignedphase += PI;
		}
	}
#ifdef DEBUG
	printf("phase alignment done to %d cells.\n",ccount);
	#ifdef COUNTUSE
		printcelluse(cim);
	#endif

#endif
}




void writecellinfo(char *fname, cimagetype *cim)
/* Write the cell info to a file. */

{
FILE *outfile;
long count;
float *outbuffer;
float *outbuffptr;
cellinfotype *cellinfoptr;


outbuffer = (float *) malloc(cim->ncells * sizeof(float));

outfile = fopen(fname,"wb");
if (outfile != NULL)
	{
	fwrite(&(cim->nxcells),sizeof(short),1,outfile);
	fwrite(&(cim->nycells),sizeof(short),1,outfile);
	fwrite(&(cim->nzcells),sizeof(short),1,outfile);
	
	outbuffptr = outbuffer;
	cellinfoptr = cim->cellinfo;
	for (count = 0; count < cim->ncells; count++)
		*outbuffptr++ = (cellinfoptr++)->scatterphase;
	fwrite(outbuffer,sizeof(float),cim->ncells,outfile);

	outbuffptr = outbuffer;
	cellinfoptr = cim->cellinfo;
	for (count = 0; count < cim->ncells; count++)
		*outbuffptr++ = (cellinfoptr++)->alignedphase;
	fwrite(outbuffer,sizeof(float),cim->ncells,outfile);
	
	outbuffptr = outbuffer;
	cellinfoptr = cim->cellinfo;
	for (count = 0; count < cim->ncells; count++)
		*outbuffptr++ = (cellinfoptr++)->mag;
	fwrite(outbuffer,sizeof(float),cim->ncells,outfile);
	
	outbuffptr = outbuffer;
	cellinfoptr = cim->cellinfo;
	for (count = 0; count < cim->ncells; count++)
		*outbuffptr++ = (cellinfoptr++)->realwt;
	fwrite(outbuffer,sizeof(float),cim->ncells,outfile);

	outbuffptr = outbuffer;
	cellinfoptr = cim->cellinfo;
	for (count = 0; count < cim->ncells; count++)
		*outbuffptr++ = (cellinfoptr++)->imagwt;
	fwrite(outbuffer,sizeof(float),cim->ncells,outfile);

	outbuffptr = outbuffer;
	cellinfoptr = cim->cellinfo;
	for (count = 0; count < cim->ncells; count++)
		*outbuffptr++ = (float)((cellinfoptr++)->border);
	fwrite(outbuffer,sizeof(float),cim->ncells,outfile);
	}
else
	printf("Error Opening File %s. \n",fname);

fclose (outfile);
free (outbuffer);

}




void writedata(char *fname, cimagetype *cim)
/* ............................................................ */
/*	Writes a 3D complex image to a data file.  The file	*/
/*	begins with 3 shorts that tell the x,y and z size of	*/
/*	the image.  These are followed by x*y*z floats that	*/
/*	are the real-part of the image, and then x*y*z more	*/
/*	floats that are the imaginary parts of the image.	*/
/*								*/
/*	INPUT:							*/
/*		fname	- File name				*/
/*		cim 	- Complex image structure		*/
/*								*/
/*	OUTPUT:							*/
/*		(file)						*/
/* ............................................................ */

{
int npts;
int count;
FILE *outfile;
short rxsize;
short rysize;
short rzsize;

outfile = fopen(fname,"wb");
if (outfile != NULL)
	{
	fwrite(&(cim->xsize),sizeof(short),1,outfile);
	fwrite(&(cim->ysize),sizeof(short),1,outfile);
	fwrite(&(cim->zsize),sizeof(short),1,outfile);

	if ( fwrite(cim->rdata, sizeof(float), cim->npts, outfile ) != cim->npts)
		printf(">>>Error writing real data.  \n");
	if ( fwrite(cim->idata, sizeof(float), cim->npts, outfile ) != cim->npts)
		printf(">>>Error writing imag data.  \n");
	}
else
	printf("Error Opening File %s. \n",fname);

fclose (outfile);

}




void makeorder(cimagetype *cim, long **cellorder, 
			short centerx, short centery, short centerz)
/* ............................................................ */
/*	Makes the order for region-growing reconstruction.	*/
/*								*/
/*	The region is grown from a cell that contains the	*/
/*	image data point expressed by (centerx,centery,centerz).*/
/*	Starting with this cell, cells are traversed in order	*/
/*	of distance from the center cell.  			*/
/*								*/
/*	This function finds the center cell, and distances of	*/
/*	other cells, and then creates the order for these 	*/
/*	cells by sorting the distances from the center cell.	*/
/*								*/
/*	INPUT:							*/
/*		cim	- Complex image data structure		*/
/*		centerx	- x-coordinate of pixel in center cell. */
/*		centery	- y-coordinate of pixel in center cell. */
/*		centerz	- z-coordinate of pixel in center cell. */
/*								*/
/*	OUTPUT:							*/
/*		cellorder - array of order for cell alignment 	*/
/*				(allocated by this function)	*/
/* ............................................................ */
{
int xccount,yccount,zccount;
short ccellx, ccelly, ccellz;
short dxx,dyy,dzz;
long *cellorderptr;
float *dist, *distptr;

long cellnum=0;


ccellx = (short) ((float)cim->nxcells * (float)centerx / (float)cim->xsize);
ccelly = (short) ((float)cim->nycells * (float)centery / (float)cim->ysize);
ccellz = (short) ((float)cim->nzcells * (float)centerz / (float)cim->zsize);

#ifdef DEBUG
	printf("Calculating Distances - center cell is (%d,%d,%d).\n",
		ccellx+1,ccelly+1,ccellz+1);
#endif

dist = (float*) malloc(cim->ncells * sizeof(float));
distptr=dist;

*cellorder = (long*) malloc(cim->ncells * sizeof(long));
cellorderptr=*cellorder;

for (zccount=0; zccount < cim->nzcells; zccount++)
	{
	dzz = (zccount-ccellz)*(zccount-ccellz);	
	for (yccount=0; yccount < cim->nycells; yccount++)
		{
		dyy = (yccount-ccelly)*(yccount-ccelly);	
		for (xccount=0; xccount < cim->nxcells; xccount++)
			{
			dxx = (xccount-ccellx)*(xccount-ccellx);	
			*distptr++ = (float)sqrt((double)(dxx+dyy+dzz));
			#ifdef DEBUG
				printf("Cell (%d,%d,%d) distance = %f\n",
						xccount+1,yccount+1,zccount+1,
					(float)sqrt((double)(dxx+dyy+dzz)));
			#endif	
			*cellorderptr++ = cellnum++;
			}
		}
	}

quicksort( *cellorder, dist, (long)0, (long)cellnum-1);

#ifdef DEBUG
	printf("\nSORTED \n\n");
	distptr=dist;
	cellorderptr=*cellorder;
	for (xccount = 0; xccount < cim->ncells; xccount++)
		{
	    	celloffset2xyz(cim, *cellorderptr, &ccellx,&ccelly,&ccellz);
		printf("Cell %d  (%d,%d,%d) distance = %f\n",
			*cellorderptr++,ccellx,ccelly,ccellz, *distptr++);
			
		}		
#endif


free(dist);


}


void readorder(char *fname, long **cellorder, long *ncells)
/* ............................................................ */
/*	Reads the order for reconstruction from a file.		*/
/*	The first number in the file is the number of cells.	*/
/*	This is followed by that many entries, all long 	*/
/*	integers.						*/
/*	The cellorder array is allocated as part of this 	*/
/*	function.						*/
/*								*/
/*	INPUT:							*/
/*		fname - file name				*/	
/*	OUTPUT:							*/
/*		cellorder - array of order for cell alignment 	*/
/*		ncells - number of cell entries read.		*/
/* ............................................................ */
{
int count;
FILE *infile;

infile = fopen(fname,"rb");
if (infile != NULL)
	{
	if (fread(ncells,sizeof(long),1,infile) != 1)
		printf("Error reading number of cell order entries.\n");

	*cellorder = (long *) malloc(*ncells * sizeof(long));
	
	if (fread(*cellorder,sizeof(long),*ncells,infile) != *ncells)
		printf("Error reading cell order entries.\n");

	}
else
	printf("Error Opening File %s. \n",fname);

fclose (infile);
}


void readdata(char *fname, cimagetype *cim, 
			short cx, short cy, short cz)
/* ............................................................ */
/*	Reads a 3D complex image from a data file.  The file	*/
/*	begins with 3 shorts that tell the x,y and z size of	*/
/*	the image.  These are followed by x*y*z floats that	*/
/*	are the real-part of the image, and then x*y*z more	*/
/*	floats that are the imaginary parts of the image.	*/
/*								*/
/*	The image data are put into a complex image structure,  */
/*	which is also set up by this routine.			*/
/*								*/
/*	INPUT:							*/
/*		fname	- File name				*/
/*		cx,cy,cz- Cell sizes in x,y,z to use.		*/
/*	OUTPUT:							*/
/*		cim 	- Complex image structure		*/
/* ............................................................ */
{
int npts;
int count;
FILE *infile;
short rxsize;
short rysize;
short rzsize;

infile = fopen(fname,"rb");
if (infile != NULL)
	{
	if (fread(&rxsize,sizeof(short),1,infile) != 1)
		printf("Error reading x-size.\n");

	if (fread(&rysize,sizeof(short),1,infile) != 1)
		printf("Error reading y-size.\n");

	if (fread(&rzsize,sizeof(short),1,infile) != 1)
		printf("Error reading z-size.\n");

	printf("Raw data is %d x %d x %d \n",rxsize,rysize,rzsize);

	cimage_allocate( cim, rxsize, rysize, rzsize, cx, cy, cz);

	npts = (rxsize)*(rysize)*(rzsize);
	if (fread(cim->rdata,sizeof(float),npts,infile) != npts)
		printf(">>> Error reading real data.\n");
	if (fread(cim->idata,sizeof(float),npts,infile) != npts)
		printf(">>> Error reading imag. data.\n");
	}
else
	printf("Error Opening File %s. \n",fname);

fclose (infile);

}





