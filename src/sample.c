
/* ............................................................ */
/*	Region-Growing Phase Correction for Balanced SSFP	*/
/*	Sample C file using functions.				*/
/*	Brian A. Hargreaves, May 2004				*/
/* ............................................................ */

/* ------------------------------------------------------------
        $log$
   ------------------------------------------------------------ */



#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "rgphcorr.h"

#ifndef PI
	#define PI 3.14159
#endif

/* #define DEBUG */

#include "rgphcorr.c"




int main (int argc, char *argv[])

{
cimagetype imdata;	/* Declare main image structure.	*/
long *cellorder;	/* Order for cell phase alignment.	*/
long ncells;
short cx=8;		/* Cell size, x */
short cy=8;		/* Cell size, y */
short cz=8;		/* Cell size, z */
short argcount;		



/* Parse Command line  -- this is mostly a sample... */

if (argc < 2)
	{
	printf("Usage:  %s <options> \n\n",argv[0]);
	printf("  (Defaults in []s)\n\n");
	printf("  -cN		- cell size [8x8x8] -> [NxNxN] \n");
	printf("  -cxN		- cell x size [8] -> N \n");
	printf("  -cyN		- cell y size [8] -> N \n");
	printf("  -czN		- cell z size [8] -> N \n");
	}

for (argcount = 1; argcount < argc; argcount++)
	{
	if ((argv[argcount][0]=='-') && (argv[argcount][1]=='c'))
		{
		if (argv[argcount][2]=='x')
			cx = atoi(&(argv[argcount][3]));
		else if (argv[argcount][2]=='y')
			cy = atoi(&(argv[argcount][3]));
		else if (argv[argcount][2]=='z')
			cz = atoi(&(argv[argcount][3]));
		else
			{
			cx = atoi(&(argv[argcount][2]));
			cy = atoi(&(argv[argcount][2]));
			cz = atoi(&(argv[argcount][2]));
			}
		}
	}


printf("---------------------------------------------------------------\n");
printf("After Parsing, Cell size = %d x %d x %d.\n",cx,cy,cz);
printf("---------------------------------------------------------------\n");



readdata("startdata", &imdata, cx,cy,cz);	/* Read data file from Matlab*/
writedata("initdata", &imdata);			/* Write data back (test)*/

print_cimagetype( &imdata );		/* Display image size stuff. */
print_stepinfotype( &imdata);		/* Display stepping stuff. */

makeorder(&imdata,&cellorder,imdata.xsize/2,imdata.ysize/2,imdata.zsize/2);


/*	This was before we had makeorder() above available.
printf("Reading ordering...\n");
readorder("orderfile", &cellorder, &ncells);
printf("Order file has %d cells.\n",ncells);
*/


printf("Doing scatter fit....\n");
dofullscatterfit( &imdata );			/* Do the scatter fit */
printf("Doing cell phase alignment...\n");

rgphasealign(&imdata, cellorder);		/* Do phase alignment */
free(cellorder);

printf("Doing cell rotation...\n");
rotateallcells( &imdata );			/* Phase-correct. */

printf("Writing Output...\n");
writecellinfo("cellinfo", &imdata);		/* Write cell info for Matlab*/
writedata("phcorrdata", &imdata);		/* Write image for Matlab */

cimage_free( &imdata);				/* Free up memory */

}






