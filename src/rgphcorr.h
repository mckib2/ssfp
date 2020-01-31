
/* ............................................................ */
/*	Blockwise Phase Correction for Balanced SSFP		*/
/*	Header File for General Functions			*/	
/*	Brian A. Hargreaves, May 2004				*/
/* ............................................................ */

/* ------------------------------------------------------------
        $log$
   ------------------------------------------------------------ */





/* ............................................................ */
/*	Cell info structure - keeps information about different */
/*	cells during the phase correction.			*/
/* ............................................................ */
typedef struct  {
	float scatterphase;	/* Fitted scatter phase angle. */
	float alignedphase;	/* Aligned phase angle.	*/
	float mag;		/* Average cell magnitude. */
	float realwt;		/* Real part of cell weight for neigbour cells*/
	float imagwt;		/* Imag part of cell weight for neigbour cells*/
	float done;		/* 0 until done, 1 when done. */
	short border;		/* 1 if cell is on an edge. */
	float dist;		/* Distance from center. */
#ifdef COUNTUSE
	short numused;		/* Number of times used by neighbours */
#endif
	} cellinfotype;	


/* ............................................................ */
/*	Step info structure - stores information about how to	*/
/*	step through the cells of an image very quickly.  This	*/
/*	probably should simply be a field of cimagetype	*/
/* ............................................................ */
typedef struct  {
	long xstep;	/* Increment when x is incremented within cell. */
	long ystep;	/* Increment when y is incremented; rewinds x.	*/
	long zstep;	/* Increment when z is incremented; rewinds x,y. */

	long cxstep;	/* Increment in image to increment to next cell x */
	long cystep;	/* Increment in image to increment to next cell y */
	long czstep;	/* Increment in image to increment to next cell z */

	long celldx;	/* Offset to increment cell number in x. */
	long celldy;	/* Offset to increment cell number in y. */
	long celldz;	/* Offset to increment cell number in z. */
	
	} stepinfotype;


/* ............................................................ */
/*	Complex Image Structure - keeps information about the   */
/*	image, such as the real/imag parts, sizes, number of	*/
/*	cells that it is divided into for correction, etc.	*/
/* ............................................................ */
typedef struct	{
	float *rdata;		/* Real-part of complex image data. */
	float *idata;		/* Imag-part of complex image data. */
	cellinfotype *cellinfo; /* Info for each cell in region-growing */
	stepinfotype *stepinfo; /* Info for stepping quickly through cells */
	short xsize;		/* image data x-size */
	short ysize;		/* image data y-size */
	short zsize;		/* image data z-size */
	long npts;		/* image data x*y*z size */
	short cxsize;		/* cell x-size */
	short cysize;		/* cell y-size */
	short czsize;		/* cell z-size */		
	long ncpts;		/* number of points in a cell */
	short nxcells;		/* number of cells, x-direction */
	short nycells;		/* number of cells, y-direction */
	short nzcells;		/* number of cells, z-direction */
	long ncells;		/* number of cells, total */
	} cimagetype;







void celloffset2xyz(cimagetype *cim, long celloffset, 
			short *cx, short *cy, short *cz);
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



int partition( long *order, float *numlist, long bot, long top);
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



void quicksort( long *order, float *numlist, long bot, long top);
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



void print_stepinfotype( cimagetype *cim);
/* ............................................................ */
/*	Function prints values of a stepinfotype structure.     */
/* ............................................................ */


	
void print_cimagetype( cimagetype *cim);
/* ............................................................ */
/*	Function prints values of a cimagetype structure.       */
/* ............................................................ */



int init_cellinfo( cimagetype *cimage);
/* ............................................................ */
/*	Initialize the information for stepping through image   */
/*	arrays and cell array.  				*/
/*								*/
/*	INPUT:							*/
/*								*/
/*	OUTPUT:							*/
/* ............................................................ */


	
int init_stepinfo( cimagetype *cimage);
/* ............................................................ */
/*	Initialize the information for stepping through image   */
/*	arrays and cell array.  				*/
/*								*/
/*	INPUT:							*/
/*								*/
/*	OUTPUT:							*/
/* ............................................................ */


void scatterfitcell(cimagetype *cim, 
			float **rptr, float **iptr, float *scatterangle,
			float *avgmagnitude);
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
/* ............................................................ */



void dofullscatterfit( cimagetype *cim);
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


void rotatecell (cimagetype *cim, float **rptr, float **iptr, 
				float rotcos, float rotsin);
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


void rotateallcells( cimagetype *cim);
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


int cimage_free( cimagetype *cim);
/* ............................................................ */
/*	Free space allocated for the complex image type.	*/
/*	size variables within the structure.			*/
/*								*/
/*	INPUT:							*/
/*		cim	- complex image structure		*/
/* ............................................................ */


int cimage_allocate( cimagetype *cim, short nx, short ny, short nz,
			short cx, short cy, short cz);
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



void setupneighbourignore(short cx, short cy, short cz,
			short ncx, short ncy, short ncz,
			short *neighbourignore);
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


#ifdef COUNTUSE
void printcelluse(cimagetype *cim);
#endif
/* ............................................................ */
/*	Print out a list of the cells, and how many times 	*/
/*	they were used as a neighbour for phase alignment.	*/
/*	This is for debugging purposes only.			*/
/* ............................................................ */



int rgphasealign(cimagetype *cim, long *cellorder);
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



void writecellinfo(char *fname, cimagetype *cim);
/* Write the cell info to a file. */


void writedata(char *fname, cimagetype *cim);
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


void makeorder(cimagetype *cim, long **cellorder, 
			short centerx, short centery, short centerz);
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


void readorder(char *fname, long **cellorder, long *ncells);
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


void readdata(char *fname, cimagetype *cim, 
			short cx, short cy, short cz);
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




