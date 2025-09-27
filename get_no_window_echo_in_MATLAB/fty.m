%%FFT in row of matrix
function fs=fty(s);
fs=fftshift(fft(fftshift(s.'))).';
% fs=fft(s.').';
