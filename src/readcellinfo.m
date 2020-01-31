
function [phsc,phal,mg,rwt,iwt,bdr] = readcellinfo(fname)

fid = fopen(fname,'rb');

s = fread(fid,3,'uint16');
phsc = fread(fid,prod(s),'float32');
phsc = reshape(phsc,s(1),s(2),s(3));

phal = fread(fid,prod(s),'float32');
phal = reshape(phal,s(1),s(2),s(3));

mg = fread(fid,prod(s),'float32');
mg = reshape(mg,s(1),s(2),s(3));

rwt = fread(fid,prod(s),'float32');
rwt = reshape(rwt,s(1),s(2),s(3));

iwt = fread(fid,prod(s),'float32');
iwt = reshape(iwt,s(1),s(2),s(3));

bdr = fread(fid,prod(s),'float32');
bdr = reshape(bdr,s(1),s(2),s(3));

fclose (fid);




