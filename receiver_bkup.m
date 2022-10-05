
%  function [decoded_data]= MyOfdmReceiver(data)

 close all;
 clear 
 %% run transmitter code to load sts and lts and other parameters 
 OFDM_TX; 
 
%% Rx processing params
% rx_data = data;  
rx_data = raw_rx_data;          % run OFDM tx code to get raw_rx_dec
LTS_CORR_THRESH = 0.8;         % Normalized threshold for LTS correlation
STS_CORR_THRESH = 0.9;
LTS_CROSS_CORR_THRESH = 0.7;
% Usage: Find all peaks whose magnitude is greater than threshold times
% the maximum magnitude after cross correlation (Packet Detection)

% Repeat the following code for each packet

%% Packet Detection
% Question (a) : Firstly, to decode we have to detect the packet and then synchronize the start of the data
% to be able to decode it. Now recall that we have 30 STS and 2.5 LTS, two sequences in the preamble of 
% our data transmission. Can we use them to do auto-correlation to detect and identify packet start? 
% If so, what to use LTS or STS? Plot the auto-correlation and print the packet start using both the techniques.

% Auto and Cross correlation of received signal with STS

length_samples = length(rx_data) - 200;
sample=length(sts_t);
output_sts_cross_corr = zeros(1,length_samples);
output_sts_auto_corr = zeros(1,length_samples);
energy = zeros(1,length_samples);
while( sample < length_samples)
    % cross-correlation with known STS
    output_sts_cross_corr(sample)= rx_data(sample-length(sts_t) + (1:length(sts_t))) * sts_t' ./norm(rx_data(sample-length(sts_t) + (1:length(sts_t)))); 
    % auto-correlation with received signal
    output_sts_auto_corr(sample) = (rx_data(sample-length(sts_t)+(1:length(sts_t)))* rx_data(sample+(1:length(sts_t)))')./norm(rx_data(sample+(1:length(sts_t))));
    % energy of signal received
    energy(sample) = (rx_data(sample+(1:length(sts_t))) * rx_data(sample+(1:length(sts_t)))') ./norm(rx_data(sample+(1:length(sts_t))));
    output_sts_auto_corr(sample) = output_sts_auto_corr(sample) ./ norm(abs(energy(sample)));
    sample= sample+1;
end
output_sts_cross_corr= output_sts_cross_corr./max(abs(output_sts_cross_corr));


%% packet detection from STS Auto-Correlation

packet_sts = zeros(1,1000);
index_sts_auto = 1;

for i = 1:1:length(output_sts_auto_corr)
    if output_sts_auto_corr(i)>STS_CORR_THRESH
        packet_sts(index_sts_auto) = i;
        index_sts_auto = index_sts_auto+1;
    end
end

peak_detected_index_sts = 0;
final_sts = 0;
flag = 0;

% counter = 0;
for i = 1:1:length(packet_sts)-1
    
    if (packet_sts(i)>0) && (packet_sts(i+1)-packet_sts(i)<=2) && (packet_sts(i)<1000)
        packet_start_sts_auto = packet_sts(i)+16;
        peak_detected_index_sts = packet_sts(i)+96+16;
        final_sts = packet_sts(i);
       
    end
    if (packet_sts(i+1)-packet_sts(i)>=2) && (flag == 0)
        disp('I am here with noise');
        peak_detected_index_sts = packet_sts(i+1);
        flag = 1;
%     else 
%         continue;
    end
end
disp('i is');
i

% for i = 1:1:length(packet_sts)
%     if packet_sts(i)
%         counter = counter+1;
%         if counter>=5
%             packet_start_sts_auto = packet_sts(i)+16;
%             peak_detected_index_sts = packet_sts(i)+96+16;
%         end
%     end
% end

%% Auto and Cross correlation of received signal with LTS

sample=length(lts_t);

% while( sample < length_samples)
%     %cross-correlation with known LTS
%     output_lts_cross_corr(sample)= rx_data(sample-length(lts_t) + (1:length(lts_t))) * lts_t' /norm(rx_data(sample+(1:length(lts_t)))); 
%     %auto-correlation with received signal
%     output_lts_auto_corr(sample) = (rx_data(sample-length(lts_t)+(1:length(lts_t)))* rx_data(sample+(1:length(lts_t)))')/norm(rx_data(sample+(1:length(lts_t))));
%     energy(sample) = (rx_data(sample+(1:length(lts_t))) * rx_data(sample+(1:length(lts_t)))') /norm(rx_data(sample+(1:length(lts_t))));
%     output_lts_auto_corr(sample) = output_lts_auto_corr(sample) / norm(abs(energy(sample)));
%     sample= sample+1;
% end
% output_lts_cross_corr = output_lts_cross_corr./max(abs(output_lts_cross_corr));

sample_idx=1;
S=lts_t;
while sample_idx<=length(rx_data)-128
    output_lts_cross_corr(sample_idx)=(rx_data(sample_idx:sample_idx+63) * S')/norm(rx_data(sample_idx:sample_idx+63));
    sample_idx=sample_idx+1;
end

%% packet detection from LTS Auto-Correlation

% packet_lts = zeros(1,1000);
% k = 1;
% peak_detected_index_lts = 0;
% for i = 1:1:length(output_lts_auto_corr)
%     if output_lts_auto_corr(i)>LTS_CORR_THRESH
%         packet_lts(k) = i;
%         k = k+1;
%         peak_detected_index_lts = i;
%     end
% end
% counter = 0;
% for i = 1:1:length(packet_lts)
%     if packet_lts(i)>0
%         counter = counter+1;
%         packet_start_lts_auto = packet_lts(i)-32-64;    % CP length = 32, LTS length = 64
%     end
% end
%% packet detection from LTS Cross-Correlation

packet_lts_cross = zeros(1,1000);
k = 1;
i = 1;
peak_detected_index_lts = 0;
for i = 1:1:length(output_lts_cross_corr)
    if abs(output_lts_cross_corr(i))>LTS_CROSS_CORR_THRESH
        packet_lts_cross(k) = i;
        k = k+1;
        peak_detected_index_lts_cross = i;
        disp('inside cross corr')
    end
end
% counter = 0;
for i = 1:1:length(packet_lts_cross)
    if packet_lts_cross(i)>0
%         counter = counter+1;
        packet_start_lts_cross = packet_lts_cross(i)-32-64;    % CP length = 32, LTS length = 64
    end
end

figure;
subplot(2,2,1)
plot(abs(output_sts_cross_corr)); 
title('STS Cross Correlation')
subplot(2,2,2)
plot(abs(output_lts_cross_corr)); 
title('LTS Cross Correlation')
subplot(2,2,3)
plot(abs(output_sts_auto_corr));
hold on ; plot(abs(energy),'r'); hold off; 
title('STS Auto Correlation')
subplot(2,2,4)
% plot(abs(output_lts_auto_corr)); 
% title('LTS Auto Correlation')

if peak_detected_index_sts ~= peak_detected_index_lts_cross
    if peak_detected_index_sts > peak_detected_index_lts_cross
        peak_detected_index = peak_detected_index_lts_cross;
    else
        peak_detected_index = peak_detected_index_lts_cross;
    end
elseif peak_detected_index_lts_cross == peak_detected_index_sts
    peak_detected_index = peak_detected_index_sts;
end

% Output: Single packet extracted from rx_data
% with knowledge of preamble (LTS) indices and payload vector indices

%% CFO estimation and correction using LTS auto correlation
% % Use two copies of LTS for cross-correlation (Reference: Thesis)
% 


%%

N=16; % since sts repeats for every 16 samples.
auto_corr = 0;
for k=0:N-1
    auto_corr = auto_corr + rx_data(122+k+N)' * rx_data(122+k);
end
coarse_cfo = angle(auto_corr);
coarse_cfo = coarse_cfo/ (2*pi*N);
coarse_cfo_op = rx_data .* exp(1j*2*pi*coarse_cfo*[0:length(rx_data)-1]);

%%
coarse_cfo_estimate = unwrap(angle(output_lts_auto_corr(peak_detected_index)))/(2*pi*(length(lts_t)));
% % coarse_cfo_estimate = angle(output_sts_auto_corr(final_sts-2))/(2*pi*(length(sts_t)));
% % coarse_cfo_estimate = 2e-04;
k = 1;
for i = 1 : 1 : length(rx_data)
    rx_data_coarse_cfo_corrected(k) = rx_data(i)*(exp(2*pi*coarse_cfo_estimate*1j*i));
    k = k+1;
end

%check CFO correction
sample=length(lts_t);

while( sample < length_samples-602)
    %cross-correlation with known LTS
    %output_lts_cross_corr(sample)= rx_data(sample-length(lts_t) + (1:length(lts_t))) * lts_t' ./norm(rx_data(sample+(1:length(lts_t)))); 
    %auto-correlation with received signal
    output_lts_auto_corr_cfo_correct(sample) = (rx_data_coarse_cfo_corrected(sample-length(lts_t)+(1:length(lts_t)))* rx_data_coarse_cfo_corrected(sample+(1:length(lts_t)))')./norm(rx_data(sample+(1:length(lts_t))));
    energy(sample) = (rx_data_coarse_cfo_corrected(sample+(1:length(lts_t))) * rx_data_coarse_cfo_corrected(sample+(1:length(lts_t)))') ./norm(rx_data_coarse_cfo_corrected(sample+(1:length(lts_t))));
    output_lts_auto_corr_cfo_correct(sample) = output_lts_auto_corr_cfo_correct(sample) ./ norm(abs(energy(sample)));
    sample= sample+1;
end

%check cfo_correction using sts 
% sample=length(sts_t);
% while( sample < length_samples-602)
%     
%     %auto-correlation with received signal
%     output_sts_auto_corr_cfo_correct(sample) = (rx_data_coarse_cfo_corrected(sample-length(sts_t)+(1:length(sts_t)))* rx_data_coarse_cfo_corrected(sample+(1:length(sts_t)))')./norm(rx_data(sample+(1:length(sts_t))));
%     energy(sample) = (rx_data_coarse_cfo_corrected(sample+(1:length(sts_t))) * rx_data_coarse_cfo_corrected(sample+(1:length(sts_t)))') ./norm(rx_data_coarse_cfo_corrected(sample+(1:length(sts_t))));
%     output_sts_auto_corr_cfo_correct(sample) = output_lts_auto_corr_cfo_correct(sample) ./ norm(abs(energy(sample)));
%     sample= sample+1;
% end

coarse_cfo_correction_check = angle(output_lts_auto_corr_cfo_correct(peak_detected_index))/(2*pi*(length(lts_t)));
% coarse_cfo_correction_check = angle(output_sts_auto_corr_cfo_correct(peak_detected_index))/(2*pi*(length(sts_t)));

k = 1;
for i = 1 : 1 : length(rx_data)
    rx_data_coarse_cfo_corrected(k) = rx_data(i)*(exp(2*pi*coarse_cfo_correction_check*1j*i));
    k = k+1;
end

figure;
plot(abs(output_lts_auto_corr_cfo_correct));
title('lts auto corr cfo corrected');


% Output: Packet with each value multiplied by CFO correction factor
%% CFO estimation method 2
sum = 0;
for m = 1:1:length(lts_t)
    sum = sum + rx_data(peak_detected_index-64+m)*conj(rx_data(peak_detected_index+m));
end


cfo_method_2 = unwrap(angle(sum))/(2*pi*64);


%% CFO estimation and correction using STS auto correlation

packet_start = peak_detected_index - length(lts_t) - (length(lts_t)/2) - (30*length(sts_t));
sample = packet_start+16;
cfo_sts_auto_corr = zeros(1,packet_start+30*length(sts_t));

while( sample < (packet_start+30*length(sts_t)))
    % auto-correlation with received signal
    output_sts_auto_corr(sample) = (rx_data(sample-length(sts_t)+(1:length(sts_t)))* rx_data(sample+(1:length(sts_t)))')./norm(rx_data(sample+(1:length(sts_t))));
    % energy of signal received
%     energy(sample) = (rx_data(sample+(1:length(sts_t))) * rx_data(sample+(1:length(sts_t)))') ./norm(rx_data(sample+(1:length(sts_t))));
%     output_sts_auto_corr(sample) = output_sts_auto_corr(sample) ./ norm(abs(energy(sample)));
    sample= sample+1;
    angle_sts_auto_corr(sample) = unwrap(angle(output_sts_auto_corr(sample)));
    cfo_sts_auto_corr(sample) = angle_sts_auto_corr(sample)/(2*pi*length(sts_t));
end

cfo_sts_auto_corr = cfo_sts_auto_corr(packet_start+16+1:end);
cfo_sts_auto_corr_mean = mean(cfo_sts_auto_corr);
% k = 1;
% for idx = 1 : 1 : length(rx_data)
%     rx_data_coarse_cfo_corrected(k) = rx_data(idx)*(exp(2*pi*cfo_sts_auto_corr_mean*1i*idx));
%     k = k+1;
% end

%% CP Removal
% Refer to the process used to add CP at TX
% Converting vector back to matrix form will help

% Output: CP free payload matrix of size (N_SC * N_OFDM_SYMS)

% Preamble removal :

rx_payload_vec = rx_data_coarse_cfo_corrected(:,(peak_detected_index+1+64:end-62));

rx_mat_with_cp = reshape(rx_payload_vec,80,500);
cp_removed_rx_payload_mat = rx_mat_with_cp(17:80,:);
figure;
plot(abs(fftshift(fft(cp_removed_rx_payload_mat))),'r');
title('fft of CP removed rx data');

%% FFT
% Refer to IFFT perfomed at TX

% Output: Symbol matrix in frequency domain of same size
rx_payload_fft = fft(cp_removed_rx_payload_mat);
figure;
plot(abs(fftshift(rx_payload_fft)));
title('rx payload before equalization');

%% Channel estimation and correction
% Use the two copies of LTS and find channel estimate (Reference: Thesis)
% Convert channel estimate to matrix form and equlaize the above matrix

% Output : Symbol equalized matrix in frequency domain of same size

channel_gain = zeros(1,N_SC);
lts_fft_seq_1 = fft(rx_data_coarse_cfo_corrected(peak_detected_index-63:peak_detected_index));
lts_fft_seq_2 = fft(rx_data_coarse_cfo_corrected(peak_detected_index+1:peak_detected_index+63+1));

% for idx = 1:1:64
%     channel_gain(idx) = (lts_fft_seq_1(idx)+lts_fft_seq_2(idx))/(2*lts_f(idx));
% end
% figure;
% plot(mag2db(abs(channel_gain)));
% title('channel gain');
% 
% channel_gain = repmat(transpose(channel_gain),1,N_OFDM_SYMS);
% rx_gain_corrected = rx_payload_fft./channel_gain;

% rx_lts1_f = fft(corrected_rx_data(698:761),64);
% rx_lts2_f = fft(corrected_rx_data(634:697),64);
H = (lts_fft_seq_1+lts_fft_seq_2)./(2*lts_f);

for i=1:N_OFDM_SYMS
    rx_gain_corrected(:,i) = rx_payload_fft(:,i)./transpose(H);
end

figure;
subplot(1,2,1)
plot(abs(fftshift(rx_gain_corrected)),'g');
title('rx gain corrected')
subplot(1,2,2)
plot(abs(fftshift(rx_payload_fft)),'r');
title('rx payload fft')

%% Advanced topics: 
%% SFO estimation and correction using pilots
% SFO manifests as a frequency-dependent phase whose slope increases
% over time as the Tx and Rx sample streams drift apart from one
% another. To correct for this effect, we calculate this phase slope at
% each OFDM symbol using the pilot tones and use this slope to
% interpolate a phase correction for each data-bearing subcarrier.

% Output: Symbol equalized matrix with pilot phase correction applied
pilot_angle = zeros(size(pilots,1),N_OFDM_SYMS);

for i = 1:length(SC_IND_PILOTS)
    pilot_angle(i,:) = (angle((rx_gain_corrected(SC_IND_PILOTS(i),:))/pilots(i)));
end


figure;
plot(abs((pilot_angle(:,31))));
title('pilot of 1st symbol');



%% Phase Error Correction using pilots
% Extract the pilots and calculate per-symbol phase error

% Output: Symbol equalized matrix with pilot phase correction applied
% Remove pilots and flatten the matrix to a vector rx_syms
% check how pilots are changing in each symbol
pilot_angle = zeros(size(pilots,1),N_OFDM_SYMS);
phase_corrected_vec = zeros(N_SC,N_OFDM_SYMS);
for i = 1:length(SC_IND_PILOTS)
    pilot_angle(i,:) = (unwrap(angle((rx_gain_corrected(SC_IND_PILOTS(i),:))/pilots(i))));
end

pilot_avg = (pilot_angle(1,:)+pilot_angle(2,:)+pilot_angle(3,:)+pilot_angle(4,:))/length(SC_IND_PILOTS);

for k = 1:N_OFDM_SYMS
    phase_corrected_vec(:,k) = rx_gain_corrected(:,k).*exp(-1i*pilot_avg(:,k));
end

figure(9);
plot(abs(pilot_angle(1,:)));
title('angle of pilots');

%% Demodulation

rx_syms = phase_corrected_vec(SC_IND_DATA, :);
rx_syms = reshape(rx_syms,1,24000);
figure;
scatter(real(rx_syms), imag(rx_syms),'filled');
title(' Signal Space of received bits');
xlabel('I'); ylabel('Q');

% rx_syms = reshape(rx_syms,1,24000);

% FEC decoder
% Demap_out = demapper(rx_syms,MOD_ORDER,1);
Demap_out = demapper(rx_syms,MOD_ORDER,sqrt(2/3*(2^MOD_ORDER-1)));

% viterbi decoder
decoded_data = vitdec(Demap_out,trel,7,'trunc','hard');

%%
% next 2 lines commented intentionally for debug
[number,ber_QPSK_1] = biterr(tx_data,decoded_data);

ber_simulation =  ber_QPSK_1
% decoded_data is the final output corresponding to tx_data, which can be used
% to calculate BER
% decoded_data = 1;
