Fs=200; %Sampling Frequency
frequency_range=[0 50]; %Limit frequencies from 0 to 25 Hz
taper_params=[3 5]; %Time bandwidth and number of tapers
window_params=[4 1]; %Window size is 4s with step size of 1s
min_nfft=0; %No minimum nfft
detrend_opt='constant'; %detrend each window by subtracting the average
weighting='unity'; %weight each taper at 1
plot_on=true; %plot spectrogram
verbose=true; %print extra info

%Generate sample chirp data
t=1/Fs:1/Fs:600; %Create 10 minutes of data
f_start=1;f_end=20; % Set chirp range in Hz
data=chirp(t,f_start,t(end),f_end,'logarithmic');

%Compute the multitaper spectrogram
[spect,stimes,sfreqs] = multitaper_spectrogram(output,Fs,frequency_range, taper_params, window_params, min_nfft, detrend_opt, weighting, plot_on, verbose);
%Or use multitaper_spectrogram_mex
% [spect,stimes,sfreqs] = multitaper_spectrogram_mex(data,Fs,frequency_range, taper_params, window_params, min_nfft, detrend_opt, weighting, plot_on, verbose);