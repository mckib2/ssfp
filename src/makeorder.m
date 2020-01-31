
function [order,coords,dist] = makeorder(datasize,cellsize,centercoord)
%function [order coords dist] = makeorder(datasize,cellsize,centercoord)
%
%	Function determines the order of cells to be processed
%	for region growing, starting from the given center coordinates.
%
%	INPUT:
%		datasize = 1x3 array, overall data size.
%		cellsize = 1x3 array, cell size.
%		centercoord = 1x3 array, data point where center cell is.
%
%	OUTPUT:
%		order = Nc x 1 array, sequential order of cells, assuming
%				cell at (1,1,1) is numbered 0.
%		coords = Nc x 3 array - coordinates of cells in order.
%		dist = distances of each cell in order from center cell.
%

ncells = datasize ./ cellsize;

if (ncells ~= floor(ncells))
	error('Cell size must evenly divide data size.');
end;


ccell = round((centercoord ./ cellsize) + .5);

cx = [1:ncells(1)]' * ones(1,ncells(2)*ncells(3));
cx = cx(:);

cy = [1:ncells(2)]' * ones(1,ncells(1)*ncells(3));
cy = reshape(cy,ncells(2),ncells(3),ncells(1));
cy = permute(cy,[3,1,2]);
cy = cy(:);

cz = [1:ncells(3)]' * ones(1,ncells(1)*ncells(2));
cz = reshape(cz,ncells(3),ncells(1),ncells(2));
cz = permute(cz,[2,3,1]);
cz = cz(:);

pos = [cx cy cz];
sep = pos - ones(prod(ncells),1)*ccell;
dist = sqrt(diag(sep * sep'));
dist = dist(:);

[y,ind] = sort(dist);

order = ind-1;
coords = pos(ind,:);
dist = dist(ind,:);


