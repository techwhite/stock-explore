import math
from math import *
import mplfinance as mpf
from matplotlib import pyplot as plt
import time
from pandas import DataFrame
import pandas as pd
from noname import *
import numpy as np
from scipy.signal import savgol_filter


pd.set_option('mode.chained_assignment', None)

# parameters
pole_min_interval = 5
min_data_day = pole_min_interval*4  # 股票历史数据 最小天数


hot_vol_ratio = 3  # 最近hot_days平均交易量/历史平均交易量
daily_change_thre = 0.5  # 最近每天价格涨幅比例
window_size = 13  # must be odd. 提高此值筛选更大周期的波段股票
ex_threshold = int(window_size / 4)  # 梯度下降提升，异常数量
half_window_size = int(window_size/2)
point_interval_thres = 2.5  #波峰、波谷之间距离和均值之间的误差阈值
point_interval_least = 4
point_change_least = 0.13

latest_pole_max_distinct = 3 # 最近的点不能太远
hot_days = 3  # 保证大于2，最近几天交易量大增，价格上涨




def angle_trunc(a):
    while a < 0.0:
        a += pi * 2
    b = a * 180 / pi
    if b > 270:
        b -= 360
    if -90 < b < 90:
        return b

    print("invalid angle:{0}", b)


def get_angle_between_points(x_orig, y_orig, x_landmark, y_landmark):
    delta_y = y_landmark - y_orig
    delta_x = x_landmark - x_orig
    return angle_trunc(atan2(delta_y, delta_x))


# angle between two vector
# v1:[x1,y1,x2,y2] director is p1 to p2
# v2:xxx
def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
    if included_angle > 180:
        included_angle = 360 - included_angle
    return included_angle


def get_trade_date(s):
    return s['trade_date']


# return: list of pair(idx, poletype)
def candidate(detail: DataFrame) -> list:
    # todo:magic
    pole_least_window = 7
    shift = pole_least_window / 2

    values = detail[Const.SMOOTH_KEY.name].tolist()

    # idx -> angle value
    point_angles = {}
    candidates = []
    for idx in range(0, len(detail) - shift):
        idx_p = values[idx]
        s = 0
        for i in range(1, shift+1):
            s += values[idx+i]
        avg_p = s/shift

        # todo: consider consolidation/increasing/decreasing mode
        # means increasing trade, fast skip
        if avg_p > idx_p:
            continue

        # means had evaluated, skip
        if idx in point_angles.keys():
            continue

        if idx == 0:
            continue

        left = 0
        if idx > shift:
            left = idx - shift
        right = idx + 2 * shift
        if right >= len(detail):
            right = len(detail)-1
        # choose minimum angle point
        minimum_angle = 360
        minimum_point = idx
        for i in range(idx, idx+shift):
            ag = angle([i, values[i], left, values[left]], [i, values[i], right, values[right]])
            point_angles[idx] = ag
            if ag < minimum_angle:
                minimum_angle = ag
                minimum_point = i
        candidates.append((minimum_point, PoleType.LOW))  # todo: low by default

    return candidates


def candidate_pole2(detail: DataFrame, meta_data: {}) -> list:
    # todo:magic
    pole_least_window = 9

    candidates = []

    # todo: need to consider consolidation
    shift = int(pole_least_window/2)
    min_p = None
    min_idx = None
    max_p = None
    max_idx = None
    pre_increase = None
    cur_increase = None
    pre_mid = None
    for idx in range(shift, len(detail), shift):
        values = detail[Const.SMOOTH_KEY.name]
        start = idx-shift
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

        cur_mid = sum(values[idx:end])/(end-idx)
        if pre_mid is None:
            pre_mid = sum(values[start:idx+1])/(idx+1-start)

        if cur_mid > pre_mid:
            cur_increase = True
        else:
            cur_increase = False
        pre_mid = cur_mid

        if max_p is None or cur_max > max_p:
            max_p = cur_max
            max_idx = cur_max_idx
        if min_p is None or cur_min < min_p:
            min_p = cur_min
            min_idx = cur_min_idx

        if pre_increase is None:
            pre_increase = cur_increase
        if cur_increase is pre_increase:
            continue

        if pre_increase is True:
            # if len(candidates) == 0:  # todo: remove for more accurate?
            #     candidates.append((min_idx, PoleType.LOW))
            #     min_idx = None
            #     min_p = None
            candidates.append((max_idx, PoleType.HIGH))
            max_idx = None
            max_p = None
        else:
            # if len(candidates) == 0:
            #     candidates.append((max_idx, PoleType.HIGH))
            #     max_idx = None
            #     max_p = None
            candidates.append((min_idx, PoleType.LOW))
            min_idx = None
            min_p = None

        pre_increase = cur_increase

    # tail process. todo：remove for more accurate?
    # if cur_increase is True and max_p is not None:
    #     candidates.append((max_idx, PoleType.HIGH))
    # if cur_increase is False and min_p is not None:
    #     candidates.append((min_idx, PoleType.LOW))

    return candidates


# return: list of pair(idx, poletype)
def candidate_pole(detail: DataFrame) -> list:
    # todo:magic
    pole_least_window = 7

    candidates = []

    # todo: need to consider consolidation
    shift = pole_least_window/2
    for idx in range(shift, len(detail)-shift):
        idx_p = detail.iloc[idx][Const.SMOOTH_KEY.name]

        is_low = True
        is_high = True
        i = 1
        while i <= shift:
            if is_low is False and is_high is False:
                break

            if is_low and (detail.iloc[idx - i][Const.SMOOTH_KEY.name] <= idx_p or
                             detail.iloc[idx + i][Const.SMOOTH_KEY.name] <= idx_p):
                is_low = False

            if is_high and (detail.iloc[idx - i][Const.SMOOTH_KEY.name] >= idx_p or
                              detail.iloc[idx + i][Const.SMOOTH_KEY.name] >= idx_p):
                is_high = False

            i += 1

        if is_low == is_high:
            continue

        pole_type = None
        if is_low:
            pole_type = PoleType.LOW
        if is_high:
            pole_type = PoleType.HIGH
        candidates.append((idx, pole_type))

    return candidates


def is_consolidation(line:str):
    return False


# principal is to check line angels between three neighboring poles a,b,c
# consolidation mode: -10(t1, can be tuned) < angle(a,b) < 10(t1), then
#   if -t2 < angle(b,c) < -90, then high pole
#   if t2 < angle(b,c) < 90, then low pole
#   if -t2 < angle(b,c) < t2, then continue consolidation
#   where t2 > 0

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


def check_discard(intervals: list, changes: list):
    avg_interval = sum(intervals) / len(intervals)
    avg_change = sum(changes) / len(changes)

    # 相比值变化，不是那么重要。但也要有底线值
    if avg_interval < point_interval_least:
        return DiscardReason.POLES_INTERVAL_TOO_SMALL

    # 高低点之间的变动不能太小
    if avg_change < point_change_least:
        return DiscardReason.POLES_CHANGE_TOO_SMALL


def get_discard_type(poles: list, detail: DataFrame):

    # todo: interval between two trade dates should be similar??

    if len(poles) < 3:
        return DiscardReason.TOO_LITTLE_POLES

    # latest pole
    pt = poles[-1][1]
    date = detail.iloc[poles[-1][0]]['trade_date']
    if pt is PoleType.HIGH:
        return DiscardReason.SHORT_SELLING

    date_dist = dict()
    for idx in range(0, len(detail)):
        date = detail.iloc[idx]['trade_date']
        date_dist[date] = idx + 1

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
            print('bad pole type in two align poles')
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

    discard_type = check_discard(intervals, changes)
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

    discard_type = check_discard(intervals, changes)
    if discard_type is not DiscardReason.NONE:
        return discard_type

    return DiscardReason.NONE


# return dataframe rows with pole type value
def find_poles(detail: DataFrame, meta_data: {}) -> list:
    poles = candidate_pole2(detail, meta_data)

    # remove_fake(poles, detail)

    return poles


def savgol_smooth(detail: DataFrame):
    # todo: tune
    orig = detail['close']

    # todo: tune
    smooth = savgol_filter(orig.values, 5, 3)

    detail[Const.SMOOTH_KEY.name] = smooth.tolist()


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
def check_quality(poles: list, detail: DataFrame, meta_data: {}) -> StockType:
    discard_reason = get_discard_type(poles, detail)

    if discard_reason is not DiscardReason.NONE:
        meta_data[DebugKey.DISCARD_REASON] = discard_reason
        meta_data[DebugKey.STOCK_TYPE] = StockType.DISCARD
        return

    meta_data[DebugKey.STOCK_TYPE] = StockType.BAND_INCREASE


# ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
def stock_analysis(stock_detail: DataFrame, meta_data: {}):
    if len(stock_detail) < min_data_day:
        meta_data[DebugKey.DISCARD_REASON] = DiscardReason.TOO_SHORT_DAY_DATA
        return StockType.DISCARD

    # todo:magic number
    # savgol_smooth(stock_detail)
    smooth_values(stock_detail, 7)
    smooth_values(stock_detail, 5)
    # smooth_values(stock_detail, 3)

    poles = find_poles(stock_detail, meta_data)
    # show_graph_pole(stock_detail, poles)

    check_quality(poles, stock_detail, meta_data)
    # if meta_data[DebugKey.STOCK_TYPE] is not StockType.DISCARD:


def date_diff(d1: str, d2: str):
    time_array1 = time.strptime(d1, "%Y%m%d")
    timestamp_day1 = int(time.mktime(time_array1))
    time_array2 = time.strptime(d2, "%Y%m%d")
    timestamp_day2 = int(time.mktime(time_array2))
    result = (timestamp_day1 - timestamp_day2) // 60 // 60 // 24
    return result


def get_new_int_value(origin: float, base: float):
    return int(origin-base)


# x is array type
# y is list of pair(data array, label name) for multiple lines drawing
def show_graph(title: str, x, y: list):
    colors = ['green', 'red', 'blue', 'yellow', 'black']
    fig = plt.figure(1)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # used to display chinese
    plt.title(title)

    plt.ylabel('y')
    plt.xlabel('x')

    for i in range(0, len(y)):
        p = y[i]
        color = colors[i]
        plt.plot(x, p[0], color=color, label=p[1])

    plt.draw()
    plt.pause(100)  # display interval before disappear
    plt.close(fig)


def show_graph_pole(detail: DataFrame, poles: list):
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

    show_kline_graph(debug_data)


def show_kline_graph(detail: DataFrame):
    ts_code = detail.iloc[0]['ts_code']

    detail = detail.rename({'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low', 'trade_date': 'Date',
                            'vol': 'Volume'}, axis='columns')

    detail['Date'] = pd.to_datetime(detail['Date'])
    detail.set_index("Date", inplace=True)
    # mpf.plot(detail, type='candle', mav=(3, 6, 9), volume=True, title=ts_code)
    mpf.plot(detail, type='candle', mav=3, volume=False, title=ts_code)


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
