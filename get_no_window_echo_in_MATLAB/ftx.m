%%FFT in column of matrix
function fs=ftx(s);
fs=fftshift(fft(fftshift(s)));
% fs=fft(s);

