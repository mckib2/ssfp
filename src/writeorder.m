
function writeorder(fname, order)

s = length(order)

fid = fopen(fname,'wb');

if (fwrite(fid,s,'int32') ~= 1) error('Error writing size.'); end;
if (fwrite(fid,order,'int32') ~= s) error('Error writing real.'); end;

fclose (fid);




