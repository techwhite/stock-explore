import mplfinance as mpf
import time
from pandas import DataFrame
import pandas as pd
from noname import *


pd.set_option('mode.chained_assignment', None)

# parameters

# better not greater than 9
pole_normal_window = 9
# todo:magic, should be smaller at the begging or end?
pole_tail_window = 3
pole_head_window = 5
shift_normal = int(pole_normal_window/2)
shift_tail = int(pole_tail_window/2)
shift_head = int(pole_head_window/2)

point_interval_least = 4
point_change_least = 0.13
min_data_day = point_interval_least*4  # 股票历史数据 最小天数
latest_pole_max_distinct = 5 # 最近的点不能太远

# used for is_sharp
hot_vol_ratio = 3  # 最近hot_days平均交易量/历史平均交易量
daily_change_thre = 0.5  # 最近每天价格涨幅比例
hot_days = 3  # 保证大于2，最近几天交易量大增，价格上涨


# 每次移动一半窗口长度的移动窗口中，判断前半部分平均值，后半部分平均值，比较两个平均值，决定是增长趋势还是下降趋势。
# 并记录最大值或最小值。若当前趋势和前一个窗口的趋势相同，则继续移动；若不同，则到达了峰点或谷底，寻找成功
# return dataframe rows with pole type value
def find_poles(detail: DataFrame, meta_data: {}) -> list:

    candidates = []

    # todo: need to consider consolidation
    values = detail[Const.SMOOTH_KEY.name]
    min_p = None
    min_idx = None
    max_p = None
    max_idx = None
    pre_increase = None
    cur_increase = None
    pre_avg = None
    idx = 0
    shift = shift_head
    while idx < len(detail):

        start = idx-shift
        if start < 0:
            start = 0
        end = idx+shift+1
        if end > len(detail):
            end = len(detail)

        cur_max = None
        cur_max_idx = None
        cur_min = None
        cur_min_idx = None
        for i in range(start, end, 1):
            if cur_max is None or values[i] > cur_max:
                cur_max = values[i]
                cur_max_idx = i
            if cur_min is None or values[i] < cur_min:
                cur_min = values[i]
                cur_min_idx = i

        cur_avg = sum(values[idx:end])/(end-idx)
        # if pre_avg is None:
        pre_avg = sum(values[start:idx+1])/(idx+1-start)

        if cur_avg > pre_avg:
            cur_increase = True
        else:
            cur_increase = False
        if pre_increase is None:
            pre_increase = cur_increase

        if max_p is None or cur_max > max_p:
            max_p = cur_max
            max_idx = cur_max_idx
        if min_p is None or cur_min < min_p:
            min_p = cur_min
            min_idx = cur_min_idx

        if cur_increase is pre_increase:
            idx = idx + shift
            continue

        if pre_increase is True:
            if max_idx <= idx:
                candidates.append((max_idx, PoleType.HIGH))
        else:
            if min_idx <= idx:
                candidates.append((min_idx, PoleType.LOW))

        min_idx = None
        min_p = None
        max_idx = None
        max_p = None

        pre_increase = cur_increase

        # 在开始和最后，需要移动的慢点，避免遗漏新的低谷或高峰
        if idx < pole_normal_window:
            shift = shift_head
        elif idx + pole_normal_window > len(detail):
            shift = shift_tail
        else:
            shift = shift_normal

        idx = idx + shift

    return candidates


def check_long_value_type(detail: DataFrame, meta_data: {}) -> list:
    values = detail[Const.SMOOTH_KEY.name]
    pre_increase = None
    cur_increase = None
    idx = 0
    shift = shift_normal
    while idx < len(detail):
        start = idx - shift
        if start < 0:
            start = 0
        end = idx + shift + 1
        if end > len(detail):
            end = len(detail)

        cur_avg = sum(values[idx:end]) / (end - idx)
        pre_avg = sum(values[start:idx+1]) / (idx+1 - start)

        if cur_avg >= pre_avg:
            cur_increase = True
        else:
            cur_increase = False
        if pre_increase is None:
            pre_increase = cur_increase

        if cur_increase is not pre_increase:
            break

        idx = idx + shift

    if cur_increase is not pre_increase:
        return

    if cur_increase is True:
        meta_data[DebugKey.STOCK_TYPE] = StockType.LONG_VALUE
    else:
        # todo: change to long value decrease type in future
        meta_data[DebugKey.STOCK_TYPE] = StockType.DISCARD
        meta_data[DebugKey.DISCARD_REASON] = DiscardReason.SHORT_SELLING


# increase mode: t1 < angle(a,b) < 90, then
#   if -t2 < angle(b,c) < -90, then high pole
# decrease mode: -t1 < angle(a,b) < -90, then
#   if  t2 < angle(b,c) < 90, if low pole
# whatever which mode, if -t2 < angle(b,c) < t2, then consolidation begin

# poles: list of pair(idx, poletype)
def remove_fake(poles: [], detail: DataFrame):
    # todo:magic
    consolidation_angle = 10

    new_poles = list()
    new_poles.append(poles[0])
    for idx in range(1, len(poles)-1):
        current = detail.iloc[poles[idx][0]]
        nxt = detail.iloc[poles[idx+1][0]]
        pre = detail.iloc[new_poles[-1][0]]
        angle_ab = get_angle_between_points(pre['trade_date'], pre[Const.SMOOTH_KEY.name],
                                            current['trade_date'], current[Const.SMOOTH_KEY.name])
        angle_bc = get_angle_between_points(current['trade_date'], current[Const.SMOOTH_KEY.name],
                                            nxt['trade_date'], nxt[Const.SMOOTH_KEY.name])

        if is_consolidation('ab'):
            # todo: consolidation need detail logic
            continue

        if is_consolidation('bc'):
            new_poles.append(poles[idx])
            continue

        if angle_ab * angle_bc < 0:
            new_poles.append(poles[idx])

    return new_poles


# todo:
def is_market_trade_increase():
    return True


def check_interval(intervals: list, changes: list):
    avg_interval = sum(intervals) / len(intervals)
    avg_change = sum(changes) / len(changes)

    # 相比值变化，不是那么重要。但也要有底线值
    if avg_interval < point_interval_least:
        return DiscardReason.POLES_INTERVAL_TOO_SMALL

    # 高低点之间的变动不能太小
    if avg_change < point_change_least:
        return DiscardReason.POLES_CHANGE_TOO_SMALL


def do_check_type(poles: list, detail: DataFrame):

    if len(poles) < 3:
        return DiscardReason.TOO_LITTLE_POLES

    # latest pole & 做空
    pt = poles[-1][1]
    if pt is PoleType.HIGH:
        # todo: fix to not discard type when ready
        # todo: return stocktype = short_selling
        return DiscardReason.SHORT_SELLING

    date = detail.iloc[poles[-1][0]]['trade_date']
    date_dist = dict()
    for idx in range(0, len(detail)):
        dt = detail.iloc[idx]['trade_date']
        date_dist[dt] = len(detail)-idx

    # 最近一个点离现在太远了
    if date_dist[date] > latest_pole_max_distinct:
        return DiscardReason.TOO_FAR

    # is_market_increase = is_market_trade_increase()
    # if is_market_increase and tp == PoleType.HIGH or is_market_increase is False and tp == PoleType.LOW:
    #     return DiscardReason.INVERSE_MARKET

    # trade date should cross between high/low points
    pre_pt = poles[0][1]
    for idx in range(1, len(poles)):
        pt = poles[idx][1]
        if pt == pre_pt:
            return DiscardReason.SAME_ALIGN_POLES

        pre_pt = pt

    # todo: optimize
    intervals = []
    changes = []
    for idx in range(0, len(poles)-1, 2):
        intervals.append(date_diff(detail.iloc[poles[idx][0]]['trade_date'],
                                   detail.iloc[poles[idx+1][0]]['trade_date']))
        changes.append(abs(detail.iloc[poles[idx][0]][Const.SMOOTH_KEY.name] -
                           detail.iloc[poles[idx+1][0]][Const.SMOOTH_KEY.name]) /
                       detail.iloc[poles[idx][0]][Const.SMOOTH_KEY.name])

    discard_type = check_interval(intervals, changes)
    if discard_type is not DiscardReason.NONE:
        return discard_type

    intervals.clear()
    changes.clear()
    for idx in range(1, len(poles) - 1, 2):
        intervals.append(date_diff(detail.iloc[poles[idx][0]]['trade_date'],
                                   detail.iloc[poles[idx+1][0]]['trade_date']))
        changes.append(abs(detail.iloc[poles[idx][0]][Const.SMOOTH_KEY.name] -
                           detail.iloc[poles[idx+1][0]][Const.SMOOTH_KEY.name]) /
                       detail.iloc[poles[idx][0]][Const.SMOOTH_KEY.name])

    discard_type = check_interval(intervals, changes)
    if discard_type is not DiscardReason.NONE:
        return discard_type

    return DiscardReason.NONE


def smooth_values(detail: DataFrame, in_size: int):
    detail[Const.SMOOTH_KEY.name] = (detail['close'] + detail['open'])/2

    shift = in_size / 2
    for idx in range(0, len(detail)):
        idx_p = detail.iloc[idx][Const.SMOOTH_KEY.name]
        left_n = 1
        left_sum = 0
        while idx-left_n >= 0 and left_n <= shift:
            left_sum += detail.iloc[idx-left_n][Const.SMOOTH_KEY.name]
            left_n += 1

        right_n = 1
        right_sum = 0
        while idx+right_n < len(detail) and right_n <= shift:
            right_sum += detail.iloc[idx+right_n][Const.SMOOTH_KEY.name]
            right_n += 1

        avg = (left_sum+right_sum)/(left_n+right_n-2)
        detail.loc[idx, Const.SMOOTH_KEY.name] = idx_p + (avg-idx_p)/2


# todo:
# poles: list of pair
def check_stock_type(poles: list, detail: DataFrame, meta_data: {}) -> StockType:

    discard_reason = do_check_type(poles, detail)
    if discard_reason is not DiscardReason.NONE:
        meta_data[DebugKey.DISCARD_REASON] = discard_reason
        meta_data[DebugKey.STOCK_TYPE] = StockType.DISCARD
        return

    meta_data[DebugKey.STOCK_TYPE] = StockType.BAND_INCREASE


# ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
def stock_analysis(stock_detail: DataFrame, meta_data: {}):
    if len(stock_detail) < min_data_day:
        meta_data[DebugKey.DISCARD_REASON] = DiscardReason.TOO_SHORT_DAY_DATA
        meta_data[DebugKey.STOCK_TYPE] = StockType.DISCARD
        return

    # todo:magic number
    # savgol_smooth(stock_detail)
    smooth_values(stock_detail, 7)
    smooth_values(stock_detail, 5)
    # smooth_values(stock_detail, 3)

    poles = find_poles(stock_detail, meta_data)

    check_stock_type(poles, stock_detail, meta_data)

    check_long_value_type(stock_detail, meta_data)

    # if meta_data[DebugKey.STOCK_TYPE] is not StockType.DISCARD:
    #     show_graph_pole(stock_detail, poles)

    show_graph_pole(stock_detail, poles, meta_data)


def date_diff(d1: str, d2: str):
    time_array1 = time.strptime(d1, "%Y%m%d")
    timestamp_day1 = int(time.mktime(time_array1))
    time_array2 = time.strptime(d2, "%Y%m%d")
    timestamp_day2 = int(time.mktime(time_array2))
    result = (timestamp_day1 - timestamp_day2) // 60 // 60 // 24
    return result


def show_graph_pole(detail: DataFrame, poles: list, meta_data: {}):
    debug_data = detail.copy(True)
    # show_kline_graph(debug_data)
    max_v = max(debug_data['high'])
    min_v = min(debug_data['low'])
    depth = max_v - min_v
    for idx in range(0, len(debug_data)):
        debug_data.loc[idx, 'open'] = debug_data.iloc[idx][Const.SMOOTH_KEY.name]
        debug_data.loc[idx, 'close'] = debug_data.iloc[idx][Const.SMOOTH_KEY.name]
        debug_data.loc[idx, 'low'] = debug_data.iloc[idx][Const.SMOOTH_KEY.name]
        debug_data.loc[idx, 'high'] = debug_data.iloc[idx][Const.SMOOTH_KEY.name]
    for elem in poles:
        if elem[1] is PoleType.LOW:
            debug_data.loc[elem[0], 'low'] = min_v - depth / 2
        else:
            debug_data.loc[elem[0], 'high'] = max_v + depth / 2

    show_kline_graph(debug_data, meta_data)


def show_kline_graph(detail: DataFrame, meta_data: {}):
    ts_code = detail.iloc[0]['ts_code']
    discard_reason = None
    st = meta_data[DebugKey.STOCK_TYPE]
    if st is StockType.DISCARD:
        discard_reason = meta_data[DebugKey.DISCARD_REASON]
    title = str(ts_code)+str('|')+str(discard_reason)+str('|')+str(st)

    detail = detail.rename({'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low', 'trade_date': 'Date',
                            'vol': 'Volume'}, axis='columns')

    detail['Date'] = pd.to_datetime(detail['Date'])
    detail.set_index("Date", inplace=True)
    # mpf.plot(detail, type='candle', mav=(3, 6, 9), volume=True, title=ts_code)
    # if discard_reason is not DiscardReason.TOO_SHORT_DAY_DATA:
    if st is not StockType.DISCARD:
        mpf.plot(detail, type='candle', mav=3, volume=False, title=title)
    # mpf.plot(detail, type='candle', mav=3, volume=False, title=title)


# ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
def is_sharp(stock_detail: DataFrame):
    if stock_detail is None:
        return False

    show_kline_graph(stock_detail)
    # show_graph(None, None)
    return False

    detail_len = len(stock_detail)
    if detail_len < min_data_day:
        return False

    # check price condition
    for idx in range(0, hot_days):
        pct_change = stock_detail.iloc[idx]['pct_chg']
        if pct_change < daily_change_thre:
            return False

        open = stock_detail.iloc[idx]['open']
        close = stock_detail.iloc[idx]['close']
        if open >= close:
            return False

        high = stock_detail.iloc[idx]['high']
        low = stock_detail.iloc[idx]['low']
        if (idx == 0 or idx == 1) and abs(open - low) >= abs(high - close):
            return False

    # check vols condition
    history_day = 30
    vols = stock_detail['vol']
    avg_vol_hot = sum(vols[0:hot_days]) / hot_days
    avg_vol_history = sum(vols[hot_days:history_day]) / history_day
    pole_vol = sum(vols[hot_days+1:hot_days])/hot_days

    if avg_vol_history == 0:
        return False

    if avg_vol_hot / avg_vol_history < hot_vol_ratio:
        return False

    if abs(avg_vol_history-pole_vol)/avg_vol_history > 0.2:
        return False

    return True
