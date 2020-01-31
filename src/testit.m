

disp('Testing rgphcorr.c');
disp('------------------');

disp('Loading data');
load samplepsssfp.mat

csize = [8 8 8];
ncells = ceil(size(dat)./csize);
dat(prod(csize.*ncells))=0;

disp('PUT CURSOR IN FAT.');
[cloc,lo,hi] = disp3dmp(dat);

disp('Saving float file.');
writedata('startdata',dat);


disp('============================================================');
!sample
disp('============================================================');

disp('Reading data.');
phcim = readdata('phcorrdata');

disp('OUTPUT IMAGE:');
disp3dmp(phcim,lo,hi);

%[phsc,phal,mg,rwt,iwt,bdr] = readcellinfo('cellinfo');



