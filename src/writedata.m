
function writedata(fname, data)

s = size(data);

fid = fopen(fname,'wb');

if (fwrite(fid,s,'uint16') ~= 3) error('Error writing size.'); end;
if (fwrite(fid,real(data),'float32') ~= prod(s)) error('Error writing real.');
end;
if (fwrite(fid,imag(data),'float32') ~= prod(s)) error('Error writing imag.');
end;

fclose (fid);




