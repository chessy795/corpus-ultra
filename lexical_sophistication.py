"""Lexical sophistication using wordfreq (D_L measure).

Based on Kyle & Crossley (2015) — Lexical Sophistication.
D_L = proportion of words in each frequency band.

Frequency bands (Zipf scale):
  Band 1: Zipf >= 6.0 (most frequent — function words)
  Band 2: Zipf 5.0-6.0 (high frequency — common content words)
  Band 3: Zipf 4.0-5.0 (mid frequency — academic words)
  Band 4: Zipf 3.0-4.0 (low frequency — rare words)
  Band 5: Zipf 2.0-3.0 (very low frequency — specialist words)
  Band 6: Zipf 1.0-2.0 (extremely low frequency)
  Band 7: Zipf < 1.0 (not in frequency list — hapax legomena)

Usage:
    from wordfreq import zipf_frequency
    tokens = ['i', 'feel', 'so', 'lonely', 'and', 'sad']
    result = compute_lexical_sophistication(tokens)
"""

from collections import Counter
from typing import Dict, List


def compute_lexical_sophistication(tokens: List[str], lang: str = 'en') -> Dict:
    """Compute lexical sophistication using wordfreq D_L measure.

    Parameters
    ----------
    tokens : list of str
        Lowercased tokens.
    lang : str
        Language code (default 'en').

    Returns
    -------
    dict with keys:
        - band_1_pct through band_7_pct: proportion in each band
        - mean_zipf: mean Zipf frequency
        - d_l: D_L score (proportion in bands 4-7, i.e., low frequency)
        - sophistication_score: weighted sophistication (band * proportion)
    """
    from wordfreq import zipf_frequency

    if not tokens:
        return {f'band_{i}_pct': 0.0 for i in range(1, 8)} | {
            'mean_zipf': 0.0, 'd_l': 0.0, 'sophistication_score': 0.0
        }

    # Classify each token into a frequency band
    bands = {i: 0 for i in range(1, 8)}
    zipf_scores = []

    for token in tokens:
        z = zipf_frequency(token, lang)
        zipf_scores.append(z)

        if z >= 6.0:
            bands[1] += 1
        elif z >= 5.0:
            bands[2] += 1
        elif z >= 4.0:
            bands[3] += 1
        elif z >= 3.0:
            bands[4] += 1
        elif z >= 2.0:
            bands[5] += 1
        elif z >= 1.0:
            bands[6] += 1
        else:
            bands[7] += 1

    total = len(tokens)
    result = {}

    # Band proportions
    for i in range(1, 8):
        result[f'band_{i}_pct'] = bands[i] / total * 100

    # Mean Zipf frequency
    result['mean_zipf'] = sum(zipf_scores) / len(zipf_scores)

    # D_L = proportion in bands 4-7 (low frequency words)
    result['d_l'] = (bands[4] + bands[5] + bands[6] + bands[7]) / total * 100

    # Weighted sophistication score (higher = more sophisticated)
    result['sophistication_score'] = sum(
        bands[i] * i for i in range(1, 8)
    ) / total

    return result


def compute_word_frequency_profile(tokens: List[str], lang: str = 'en') -> Dict[str, float]:
    """Compute word frequency profile — how frequent are the words used?

    Returns the mean Zipf frequency and standard deviation.
    """
    from wordfreq import zipf_frequency

    if not tokens:
        return {'mean_zipf': 0.0, 'std_zipf': 0.0, 'median_zipf': 0.0}

    zipf_scores = [zipf_frequency(t, lang) for t in tokens]
    mean_z = sum(zipf_scores) / len(zipf_scores)
    std_z = (sum((z - mean_z) ** 2 for z in zipf_scores) / len(zipf_scores)) ** 0.5
    sorted_z = sorted(zipf_scores)
    median_z = sorted_z[len(sorted_z) // 2]

    return {
        'mean_zipf': mean_z,
        'std_zipf': std_z,
        'median_zipf': median_z,
        'min_zipf': min(zipf_scores),
        'max_zipf': max(zipf_scores),
    }
