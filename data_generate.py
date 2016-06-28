#!/usr/bin/env python2
from gnuradio import gr
from gnuradio import audio, analog
from gnuradio import digital
from gnuradio import blocks
from grc_gnuradio import blks2 as grc_blks2
import inspector
import threading
import time
import numpy

snr = ["20","15","10","5","0","-5","-10","-15","-20"] 
snrv = [[1,0.32],[1,0.435],[1,0.56],[1,0.75],[1,1],[0.75,1],[0.56,1],[0.435,1],[0.32,1]]

class my_top_block(gr.top_block):
    def __init__(self,modulation,snr):

        self.samp_rate = samp_rate = 32000
        gr.top_block.__init__(self)

        self.inspector_scf_0 = inspector.scf()

        if modulation == "2psk":
            self.digital_mod = digital.psk.psk_mod(
                constellation_points=2,
                mod_code="gray",
                differential=True,
                samples_per_symbol=2,
                excess_bw=0.35,
                verbose=False,
                log=False,
            )
            bits = 1
        elif modulation == "4psk":
            self.digital_mod = digital.psk.psk_mod(
                constellation_points=4,
                mod_code="gray",
                differential=True,
                samples_per_symbol=2,
                excess_bw=0.35,
                verbose=False,
                log=False,
            )
            bits = 2
        elif modulation == "8psk":
            self.digital_mod = digital.psk.psk_mod(
                constellation_points=8,
                mod_code="gray",
                differential=True,
                samples_per_symbol=2,
                excess_bw=0.35,
                verbose=False,
                log=False,
            )
            bits = 3
        elif modulation == "fsk":
            self.digital_mod = digital.gfsk_mod(
        	    samples_per_symbol=2,
        	    sensitivity=1.0,
        	    bt=0.35,
        	    verbose=False,
        	    log=False,
            )
            bits = 1

        self.blocks_add_xx_1 = blocks.add_vcc(1)
        self.blocks_multiply_const_vxx_3 = blocks.multiply_const_vcc((snrv[snr][0], ))
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 1024)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_float*1)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, snrv[snr][1], 0)

        self.analog_random_source_x_0 = blocks.vector_source_b(map(int, numpy.random.randint(0, 256, 10000)), False)

        self.blks2_packet_encoder_0 = grc_blks2.packet_mod_b(grc_blks2.packet_encoder(
        		samples_per_symbol=2,
        		bits_per_symbol=bits,
        		preamble="",
        		access_code="",
        		pad_for_usrp=True,
        	),
        	payload_length=90,
        )

        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, "data/train-2/%s-snr%d.dat" % (modulation,snr), False)

        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_1, 1))    
        self.connect((self.blocks_multiply_const_vxx_3, 0), (self.blocks_add_xx_1, 0))    
        self.connect((self.analog_random_source_x_0, 0), (self.blks2_packet_encoder_0, 0))  
        self.connect((self.blks2_packet_encoder_0, 0), (self.digital_mod, 0)) 
        self.connect((self.digital_mod, 0), (self.blocks_throttle_0, 0))   
        self.connect((self.blocks_throttle_0, 0),(self.blocks_multiply_const_vxx_3, 0))    
        self.connect((self.blocks_add_xx_1, 0),(self.blocks_file_sink_0, 0))    

if __name__ == '__main__':
    try:
        for m in ["2psk","4psk","8psk","fsk"]:

            for snr in range(0,9):
                tb = my_top_block(m,snr)
                tb.start()
                time.sleep(20)
                tb.stop()
    except [[KeyboardInterrupt]]:
        pass

