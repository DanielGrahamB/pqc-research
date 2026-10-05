"""IoT cost views of measured results. Arithmetic on saved thresholds and keys; no new simulation."""
import math

UMI_NLOS_SLOPE_DB = 35.3   # TR 38.901 UMi NLOS path loss, dB per decade of distance
UE_MAX_DBM = 23.           # 3GPP power class 3 UE
PAIRS = {'ggh_full_uncoded':'none_uncoded','ggh_subband_uncoded':'none_uncoded',
         'ggh_full_ldpc':'none_ldpc','ggh_subband_ldpc':'none_ldpc'}
MATCH = ('experiment','scenario','K','M','modulation','receiver','csi','csi_nmse_db','tx_evm_db','adc_bits')
# Published sizes for comparison (FIPS 203 ML-KEM-512; AES-128 key).
REFERENCE_KEYS = [{'scheme':'ML-KEM-512','public_key_bytes':800,'ciphertext_bytes_per_key':768},
                  {'scheme':'AES-128','public_key_bytes':0,'secret_key_bytes':16}]

def _threshold(rows):
    """Required Eb/N0 at BLER=1e-2, or a bound when the sweep did not bracket it."""
    r = rows[0]
    value = r.get('required_ebno_bler_1e2')
    if value is not None and value == value: return float(value), 'measured'
    if r.get('threshold_status') == 'below_sweep_range': return min(float(x['ebno_db']) for x in rows), 'upper_bound'
    return max(float(x['ebno_db']) for x in rows), 'lower_bound'

def penalty_table(rows):
    """Pair every GGH curve with its plain reference (same coding); express the SNR gap as IoT cost."""
    curves = {}
    for r in rows: curves.setdefault((r['experiment'], r['config_index']), []).append(r)
    plain = {}
    for items in curves.values():
        r = items[0]
        if r['scheme'] in PAIRS.values(): plain[(r['scheme'],)+tuple(str(r.get(k)) for k in MATCH)] = _threshold(items)
    table = []
    for items in curves.values():
        r = items[0]
        if r['scheme'] not in PAIRS: continue
        ref = plain.get((PAIRS[r['scheme']],)+tuple(str(r.get(k)) for k in MATCH))
        if ref is None: continue
        ggh, status = _threshold(items)
        gap = ggh - ref[0]
        bits = int(math.log2(int(str(r['modulation']).split('-')[0])))
        esno = ggh + 10*math.log10(bits*float(r['coderate']))
        table.append({**{k: r.get(k) for k in MATCH}, 'scheme': r['scheme'], 'tile': r.get('tile'),
                      'plain_ebno_db': ref[0], 'ggh_ebno_db': ggh, 'ggh_status': status,
                      'penalty_db': gap, 'penalty_is': '>=' if status == 'lower_bound' else '=',
                      'ue_power_needed_dbm': UE_MAX_DBM + gap,
                      'range_divided_by': 10**(gap/UMI_NLOS_SLOPE_DB),
                      'adc_bits_needed': math.ceil((esno - 1.76)/6.02)})
    return table

def key_costs(key, bits_per_symbol=4):
    """Storage and operations per information bit (uncoded) for one GGH key."""
    n = key.n
    entry_bits = math.ceil(math.log2(float(key.B.abs().max()) + 1)) + 1   # signed integer entries
    return {'scheme': f'GGH n={n}', 'public_key_bytes_float64': key.metadata['public_key_bytes'],
            'public_key_bytes_packed': math.ceil(n*n*entry_bits/8), 'private_key_bytes_float64': key.metadata['private_key_bytes'],
            'keygen_ms': key.metadata['keygen_ms'],
            'encrypt_mac_per_bit': 2*n/bits_per_symbol, 'decrypt_mac_per_bit': 4*n/bits_per_symbol}
