
function dat = readdata(fname)

fid = fopen(fname,'rb');

s = fread(fid,3,'uint16');
dat = fread(fid,prod(s),'float32');
dat = dat + i*fread(fid,prod(s),'float32');

fclose (fid);

dat = reshape(dat,s(1),s(2),s(3));



