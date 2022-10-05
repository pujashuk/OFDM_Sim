close all;
clear

%% test simulation 
OFDM_TX; % run base transmitter which you can use for developing. 

[decoded_data]= MyOfdmReceiver(raw_rx_data);
 
% rx_data is the final output corresponding to tx_data, which can be used
% to calculate BER
%%
% next 2 lines commented intentionally for debug
[number,ber_QPSK_1] = biterr(tx_data,decoded_data);

ber_simulation =  ber_QPSK_1


%% test real world dat 
load('tx_data_QPSK.mat','data') % contain tx data bits 
tx_data= data;
load('packet_set_QPSK.mat','data'); % contains captured data using USRP in variable data
 
[decoded_data]= MyOfdmReceiver(data);
 
% rx_data is the final output corresponding to tx_data, which can be used
% to calculate BER
[number,ber_QPSK_1] = biterr(tx_data,decoded_data);

ber_QPSK_1

