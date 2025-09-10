"""
Technical analysis indicators for stock trading algorithms.

Provides comprehensive technical indicators including moving averages, oscillators,
momentum indicators, volatility measures, and volume analysis tools.
"""

from typing import List, Tuple, Dict, Optional, NamedTuple
import math
from ..pure.ops import mean, median, std as stdev, var as variance

# Type definitions
PriceData = List[float]
VolumeData = List[float]
OHLCData = List[Tuple[float, float, float, float]]  # (open, high, low, close)

class IndicatorResult(NamedTuple):
    """Result container for technical indicators."""
    values: List[float]
    signals: List[int]  # -1: sell, 0: hold, 1: buy

class BollingerBandsResult(NamedTuple):
    """Bollinger Bands result container."""
    upper: List[float]
    middle: List[float]
    lower: List[float]
    percent_b: List[float]

class MACDResult(NamedTuple):
    """MACD result container."""
    macd_line: List[float]
    signal_line: List[float]
    histogram: List[float]


# MOVING AVERAGES

def simple_moving_average(prices: PriceData, period: int = 20) -> List[float]:
    """Calculate Simple Moving Average (SMA)."""
    if period <= 0 or period > len(prices):
        return []
    
    sma = []
    for i in range(len(prices)):
        if i < period - 1:
            sma.append(None)
        else:
            avg = sum(prices[i - period + 1:i + 1]) / period
            sma.append(avg)
    
    return sma


def exponential_moving_average(prices: PriceData, period: int = 20, 
                             alpha: Optional[float] = None) -> List[float]:
    """Calculate Exponential Moving Average (EMA)."""
    if not prices or period <= 0:
        return []
    
    if alpha is None:
        alpha = 2.0 / (period + 1)
    
    ema = [prices[0]]  # Start with first price
    
    for i in range(1, len(prices)):
        ema_val = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        ema.append(ema_val)
    
    return ema


def weighted_moving_average(prices: PriceData, period: int = 20) -> List[float]:
    """Calculate Weighted Moving Average (WMA)."""
    if period <= 0 or period > len(prices):
        return []
    
    wma = []
    weights = list(range(1, period + 1))
    weight_sum = sum(weights)
    
    for i in range(len(prices)):
        if i < period - 1:
            wma.append(None)
        else:
            weighted_sum = sum(prices[i - period + 1 + j] * weights[j] for j in range(period))
            wma.append(weighted_sum / weight_sum)
    
    return wma


def hull_moving_average(prices: PriceData, period: int = 20) -> List[float]:
    """Calculate Hull Moving Average (HMA) - reduces lag while maintaining smoothness."""
    if period <= 0 or len(prices) < period:
        return []
    
    # Calculate WMA(n/2) and WMA(n)
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(math.sqrt(period)))
    
    wma_half = weighted_moving_average(prices, half_period)
    wma_full = weighted_moving_average(prices, period)
    
    # Calculate 2*WMA(n/2) - WMA(n)
    diff_series = []
    for i in range(len(prices)):
        if wma_half[i] is not None and wma_full[i] is not None:
            diff_series.append(2 * wma_half[i] - wma_full[i])
        else:
            diff_series.append(None)
    
    # Apply WMA to the difference series
    valid_diffs = [x for x in diff_series if x is not None]
    if len(valid_diffs) < sqrt_period:
        return diff_series
    
    return weighted_moving_average(valid_diffs, sqrt_period)


def kaufman_adaptive_moving_average(prices: PriceData, period: int = 20) -> List[float]:
    """Calculate Kaufman's Adaptive Moving Average (KAMA)."""
    if period <= 0 or len(prices) < period:
        return []
    
    kama = [prices[0]]  # Start with first price
    fastest_sc = 2.0 / (2 + 1)    # Fastest smoothing constant
    slowest_sc = 2.0 / (30 + 1)   # Slowest smoothing constant
    
    for i in range(1, len(prices)):
        if i < period:
            kama.append(prices[i])
            continue
        
        # Calculate efficiency ratio
        change = abs(prices[i] - prices[i - period])
        volatility = sum(abs(prices[j] - prices[j - 1]) for j in range(i - period + 1, i + 1))
        
        if volatility == 0:
            efficiency = 0
        else:
            efficiency = change / volatility
        
        # Calculate smoothing constant
        sc = (efficiency * (fastest_sc - slowest_sc) + slowest_sc) ** 2
        
        # Calculate KAMA
        kama_val = kama[i - 1] + sc * (prices[i] - kama[i - 1])
        kama.append(kama_val)
    
    return kama


# OSCILLATORS

def rsi(prices: PriceData, period: int = 14) -> IndicatorResult:
    """Calculate Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return IndicatorResult([], [])
    
    gains = []
    losses = []
    
    # Calculate price changes
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-change)
    
    if len(gains) < period:
        return IndicatorResult([], [])
    
    # Calculate initial average gain and loss
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    rsi_values = []
    
    # Calculate RSI for each period
    for i in range(period, len(gains)):
        # Wilder's smoothing
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi_val = 100
        else:
            rs = avg_gain / avg_loss
            rsi_val = 100 - (100 / (1 + rs))
        
        rsi_values.append(rsi_val)
    
    # Generate signals
    signals = []
    for rsi_val in rsi_values:
        if rsi_val > 70:
            signals.append(-1)  # Overbought - sell signal
        elif rsi_val < 30:
            signals.append(1)   # Oversold - buy signal
        else:
            signals.append(0)   # Hold
    
    return IndicatorResult(rsi_values, signals)


def stochastic_oscillator(ohlc_data: OHLCData, k_period: int = 14, 
                         d_period: int = 3) -> Dict[str, List[float]]:
    """Calculate Stochastic Oscillator (%K and %D)."""
    if len(ohlc_data) < k_period:
        return {'%K': [], '%D': []}
    
    k_values = []
    
    for i in range(k_period - 1, len(ohlc_data)):
        # Get highest high and lowest low over period
        period_data = ohlc_data[i - k_period + 1:i + 1]
        highest_high = max(candle[1] for candle in period_data)  # High
        lowest_low = min(candle[2] for candle in period_data)    # Low
        current_close = ohlc_data[i][3]  # Close
        
        if highest_high == lowest_low:
            k_val = 50  # Avoid division by zero
        else:
            k_val = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        k_values.append(k_val)
    
    # Calculate %D as SMA of %K
    d_values = simple_moving_average(k_values, d_period)
    d_values = [x for x in d_values if x is not None]
    
    return {'%K': k_values, '%D': d_values}


def williams_r(ohlc_data: OHLCData, period: int = 14) -> List[float]:
    """Calculate Williams %R."""
    if len(ohlc_data) < period:
        return []
    
    williams_r_values = []
    
    for i in range(period - 1, len(ohlc_data)):
        period_data = ohlc_data[i - period + 1:i + 1]
        highest_high = max(candle[1] for candle in period_data)
        lowest_low = min(candle[2] for candle in period_data)
        current_close = ohlc_data[i][3]
        
        if highest_high == lowest_low:
            wr_val = -50
        else:
            wr_val = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
        
        williams_r_values.append(wr_val)
    
    return williams_r_values


def commodity_channel_index(ohlc_data: OHLCData, period: int = 20) -> List[float]:
    """Calculate Commodity Channel Index (CCI)."""
    if len(ohlc_data) < period:
        return []
    
    # Calculate typical prices
    typical_prices = []
    for candle in ohlc_data:
        typical_price = (candle[1] + candle[2] + candle[3]) / 3  # (H + L + C) / 3
        typical_prices.append(typical_price)
    
    cci_values = []
    
    for i in range(period - 1, len(typical_prices)):
        period_prices = typical_prices[i - period + 1:i + 1]
        sma_tp = sum(period_prices) / period
        
        # Calculate mean deviation
        mean_deviation = sum(abs(price - sma_tp) for price in period_prices) / period
        
        if mean_deviation == 0:
            cci_val = 0
        else:
            cci_val = (typical_prices[i] - sma_tp) / (0.015 * mean_deviation)
        
        cci_values.append(cci_val)
    
    return cci_values


# MOMENTUM INDICATORS

def macd(prices: PriceData, fast_period: int = 12, slow_period: int = 26, 
         signal_period: int = 9) -> MACDResult:
    """Calculate Moving Average Convergence Divergence (MACD)."""
    if len(prices) < slow_period:
        return MACDResult([], [], [])
    
    # Calculate EMAs
    ema_fast = exponential_moving_average(prices, fast_period)
    ema_slow = exponential_moving_average(prices, slow_period)
    
    # Calculate MACD line
    macd_line = []
    for i in range(len(prices)):
        if i >= slow_period - 1:
            macd_val = ema_fast[i] - ema_slow[i]
            macd_line.append(macd_val)
    
    # Calculate signal line (EMA of MACD)
    signal_line = exponential_moving_average(macd_line, signal_period)
    
    # Calculate histogram
    histogram = []
    for i in range(len(signal_line)):
        if i < len(macd_line) and signal_line[i] is not None:
            hist_val = macd_line[i] - signal_line[i]
            histogram.append(hist_val)
    
    return MACDResult(macd_line, signal_line, histogram)


def momentum(prices: PriceData, period: int = 10) -> List[float]:
    """Calculate Price Momentum."""
    if len(prices) < period:
        return []
    
    momentum_values = []
    
    for i in range(period, len(prices)):
        mom_val = prices[i] - prices[i - period]
        momentum_values.append(mom_val)
    
    return momentum_values


def rate_of_change(prices: PriceData, period: int = 10) -> List[float]:
    """Calculate Rate of Change (ROC)."""
    if len(prices) < period:
        return []
    
    roc_values = []
    
    for i in range(period, len(prices)):
        if prices[i - period] == 0:
            roc_val = 0
        else:
            roc_val = ((prices[i] - prices[i - period]) / prices[i - period]) * 100
        roc_values.append(roc_val)
    
    return roc_values


def awesome_oscillator(ohlc_data: OHLCData, fast_period: int = 5, 
                      slow_period: int = 34) -> List[float]:
    """Calculate Awesome Oscillator (AO)."""
    if len(ohlc_data) < slow_period:
        return []
    
    # Calculate median prices (H + L) / 2
    median_prices = []
    for candle in ohlc_data:
        median_price = (candle[1] + candle[2]) / 2
        median_prices.append(median_price)
    
    # Calculate SMAs
    sma_fast = simple_moving_average(median_prices, fast_period)
    sma_slow = simple_moving_average(median_prices, slow_period)
    
    # Calculate AO
    ao_values = []
    for i in range(len(median_prices)):
        if sma_fast[i] is not None and sma_slow[i] is not None:
            ao_val = sma_fast[i] - sma_slow[i]
            ao_values.append(ao_val)
    
    return ao_values


# VOLATILITY INDICATORS

def bollinger_bands(prices: PriceData, period: int = 20, 
                   std_dev: float = 2.0) -> BollingerBandsResult:
    """Calculate Bollinger Bands."""
    if len(prices) < period:
        return BollingerBandsResult([], [], [], [])
    
    sma = simple_moving_average(prices, period)
    upper_bands = []
    lower_bands = []
    percent_b = []
    
    for i in range(len(prices)):
        if sma[i] is not None:
            # Calculate standard deviation for the period
            period_data = prices[i - period + 1:i + 1]
            std = stdev(period_data) if len(period_data) > 1 else 0
            
            upper = sma[i] + (std_dev * std)
            lower = sma[i] - (std_dev * std)
            
            upper_bands.append(upper)
            lower_bands.append(lower)
            
            # Calculate %B
            if upper != lower:
                pb = (prices[i] - lower) / (upper - lower)
            else:
                pb = 0.5
            percent_b.append(pb)
        else:
            upper_bands.append(None)
            lower_bands.append(None)
            percent_b.append(None)
    
    middle_bands = sma
    return BollingerBandsResult(upper_bands, middle_bands, lower_bands, percent_b)


def average_true_range(ohlc_data: OHLCData, period: int = 14) -> List[float]:
    """Calculate Average True Range (ATR)."""
    if len(ohlc_data) < 2:
        return []
    
    true_ranges = []
    
    for i in range(1, len(ohlc_data)):
        high = ohlc_data[i][1]
        low = ohlc_data[i][2]
        prev_close = ohlc_data[i - 1][3]
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)
    
    # Calculate ATR using EMA
    return exponential_moving_average(true_ranges, period)


def keltner_channels(ohlc_data: OHLCData, period: int = 20, 
                    multiplier: float = 2.0) -> Dict[str, List[float]]:
    """Calculate Keltner Channels."""
    if len(ohlc_data) < period:
        return {'upper': [], 'middle': [], 'lower': []}
    
    # Calculate typical prices and ATR
    typical_prices = [(candle[1] + candle[2] + candle[3]) / 3 for candle in ohlc_data]
    atr_values = average_true_range(ohlc_data, period)
    
    # Calculate EMA of typical prices
    ema_tp = exponential_moving_average(typical_prices, period)
    
    upper_channel = []
    lower_channel = []
    
    for i in range(len(ema_tp)):
        if ema_tp[i] is not None and i < len(atr_values) and atr_values[i] is not None:
            upper = ema_tp[i] + (multiplier * atr_values[i])
            lower = ema_tp[i] - (multiplier * atr_values[i])
            upper_channel.append(upper)
            lower_channel.append(lower)
        else:
            upper_channel.append(None)
            lower_channel.append(None)
    
    return {'upper': upper_channel, 'middle': ema_tp, 'lower': lower_channel}


# VOLUME INDICATORS

def on_balance_volume(prices: PriceData, volumes: VolumeData) -> List[float]:
    """Calculate On-Balance Volume (OBV)."""
    if len(prices) != len(volumes) or len(prices) < 2:
        return []
    
    obv_values = [0]  # Start with 0
    
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            obv_val = obv_values[i - 1] + volumes[i]
        elif prices[i] < prices[i - 1]:
            obv_val = obv_values[i - 1] - volumes[i]
        else:
            obv_val = obv_values[i - 1]
        
        obv_values.append(obv_val)
    
    return obv_values


def volume_weighted_average_price(ohlc_data: OHLCData, volumes: VolumeData) -> List[float]:
    """Calculate Volume Weighted Average Price (VWAP)."""
    if len(ohlc_data) != len(volumes):
        return []
    
    vwap_values = []
    cumulative_pv = 0
    cumulative_volume = 0
    
    for i, (candle, volume) in enumerate(zip(ohlc_data, volumes)):
        typical_price = (candle[1] + candle[2] + candle[3]) / 3  # (H + L + C) / 3
        pv = typical_price * volume
        
        cumulative_pv += pv
        cumulative_volume += volume
        
        if cumulative_volume == 0:
            vwap = typical_price
        else:
            vwap = cumulative_pv / cumulative_volume
        
        vwap_values.append(vwap)
    
    return vwap_values


def money_flow_index(ohlc_data: OHLCData, volumes: VolumeData, 
                    period: int = 14) -> List[float]:
    """Calculate Money Flow Index (MFI)."""
    if len(ohlc_data) != len(volumes) or len(ohlc_data) < period + 1:
        return []
    
    # Calculate typical prices and money flow
    typical_prices = []
    money_flows = []
    
    for i, (candle, volume) in enumerate(zip(ohlc_data, volumes)):
        typical_price = (candle[1] + candle[2] + candle[3]) / 3
        typical_prices.append(typical_price)
        
        if i > 0:
            money_flow = typical_price * volume
            if typical_price > typical_prices[i - 1]:
                money_flows.append(('positive', money_flow))
            elif typical_price < typical_prices[i - 1]:
                money_flows.append(('negative', money_flow))
            else:
                money_flows.append(('neutral', money_flow))
    
    mfi_values = []
    
    for i in range(period - 1, len(money_flows)):
        period_flows = money_flows[i - period + 1:i + 1]
        
        positive_flow = sum(flow[1] for flow in period_flows if flow[0] == 'positive')
        negative_flow = sum(flow[1] for flow in period_flows if flow[0] == 'negative')
        
        if negative_flow == 0:
            mfi = 100
        else:
            money_ratio = positive_flow / negative_flow
            mfi = 100 - (100 / (1 + money_ratio))
        
        mfi_values.append(mfi)
    
    return mfi_values


# Z-SCORE INDICATOR

def z_score(prices: PriceData, period: int = 20) -> IndicatorResult:
    """
    Calculate Z-Score for mean reversion trading.
    
    Z-Score measures how many standard deviations the current price
    is away from the mean price over a lookback period.
    
    Args:
        prices: Price data
        period: Lookback period for calculating mean and std
        
    Returns:
        IndicatorResult with z-scores and trading signals
    """
    if len(prices) < period:
        return IndicatorResult([], [])
    
    z_scores = []
    signals = []
    
    for i in range(period - 1, len(prices)):
        period_prices = prices[i - period + 1:i + 1]
        mean_price = sum(period_prices) / period
        
        if len(period_prices) > 1:
            std_price = stdev(period_prices)
        else:
            std_price = 0
        
        if std_price == 0:
            z_score_val = 0
        else:
            z_score_val = (prices[i] - mean_price) / std_price
        
        z_scores.append(z_score_val)
        
        # Generate trading signals
        if z_score_val > 2.0:
            signals.append(-1)  # Sell signal (price too high)
        elif z_score_val < -2.0:
            signals.append(1)   # Buy signal (price too low)
        elif abs(z_score_val) < 0.5:
            signals.append(0)   # Close position (near mean)
        else:
            signals.append(0)   # Hold current position
    
    return IndicatorResult(z_scores, signals)