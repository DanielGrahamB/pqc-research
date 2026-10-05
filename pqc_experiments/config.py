from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    scenario: str = 'umi'
    num_users: int = 4
    num_bs_ant: int = 8
    bits_per_symbol: int = 4
    coderate: float = 1.0
    coded: bool = False
    receiver: str = 'lmmse'
    csi_mode: str = 'perfect'
    csi_nmse_db: float = -20.0
    security_mode: str = 'none'
    lattice_n: int = 512
    tile_width: int = 128
    tile_time: int = 4
    true_power_normalization: bool = True
    seed: int = 20
    batch_size: int = 8
    fft_size: int = 128
    num_data_intervals: int = 16
    pilot_indices: tuple = (2, 11)
    cyclic_prefix_length: int = 20
    subcarrier_spacing: float = 30e3
    carrier_frequency: float = 3.5e9
    ldpc_iterations: int = 20
    hard_llr_crossover: float = 0.05
    device: str = 'cpu'
    enable_pathloss: bool = False
    enable_shadow_fading: bool = False
    tx_evm_db: float = None
    adc_bits: int = None

    def __post_init__(self):
        for name, choices in [('scenario', ('umi','uma','rma')), ('receiver', ('zf','lmmse','mmse_sic')), ('csi_mode', ('perfect','ls_nn','ls_lin','controlled_nmse')), ('security_mode', ('none','ggh_full','ggh_subband'))]:
            if getattr(self,name) not in choices: raise ValueError(f'{name}: expected {choices}')
        if self.bits_per_symbol not in (2,4,6,8): raise ValueError('Use square QPSK/16-QAM/64-QAM/256-QAM')
        if self.tx_evm_db is not None and self.tx_evm_db >= 0: raise ValueError('Transmit EVM must be negative dB')
        if self.adc_bits is not None and not 1 <= self.adc_bits <= 24: raise ValueError('ADC bits must be in 1..24')
        if min(self.num_users,self.num_bs_ant,self.batch_size,self.fft_size,self.num_data_intervals,self.tile_width,self.tile_time) < 1: raise ValueError('Dimensions must be positive')
        if self.num_bs_ant < self.num_users: raise ValueError('This receiver comparison requires M >= K')
        if self.fft_size % self.num_users: raise ValueError('Kronecker pilots require fft_size divisible by K')
        if self.num_bs_ant % 2: raise ValueError('Dual-polarized BS requires even M')
        if not 0 < self.coderate <= 1 or (not self.coded and self.coderate != 1): raise ValueError('Uncoded rate must be 1; coded rate in (0,1)')
        if self.coded and self.coderate >= 0.95: raise ValueError('5G LDPC requires rate below .95')
        if self.security_mode != 'none':
            if self.tile_width*self.tile_time != self.lattice_n: raise ValueError('Tile area must equal lattice_n')
            if self.fft_size % self.tile_width or self.num_data_intervals % self.tile_time: raise ValueError('Tiles must partition data grid without padding')
            if self.security_mode == 'ggh_full' and self.tile_width != self.fft_size: raise ValueError('Full-band tile must span FFT')
            if self.security_mode == 'ggh_subband' and self.tile_width >= self.fft_size: raise ValueError('Subband tile must localize frequency')
        if len(set(self.pilot_indices)) != len(self.pilot_indices) or not self.pilot_indices or min(self.pilot_indices)<0 or max(self.pilot_indices)>=self.num_symbols: raise ValueError('Invalid pilot intervals')
        if not 0 < self.hard_llr_crossover < .5: raise ValueError('BSC crossover must be in (0,.5)')

    @property
    def num_symbols(self): return self.num_data_intervals + len(self.pilot_indices)
    @property
    def n_bits(self): return self.fft_size*self.num_data_intervals*self.bits_per_symbol
    @property
    def k_bits(self): return int(self.n_bits*self.coderate) if self.coded else self.n_bits
    @property
    def effective_rate(self): return self.k_bits/self.n_bits
    @property
    def scheme(self): return self.security_mode + ('_ldpc' if self.coded else '_uncoded')
