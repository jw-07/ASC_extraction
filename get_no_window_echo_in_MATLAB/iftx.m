%%IFFT in column of matrix
function s=iftx(fs);
s=fftshift(ifft(fftshift(fs)));
% s=ifft(fs);
