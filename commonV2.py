import mplfinance as mpf
import time
import numpy as np
from pandas import DataFrame
import pandas as pd
from noname import *
from math import *

import matplotlib.pyplot as plt

pd.set_option('mode.chained_assignment', None)

# 指定一个或多个stocktype类型，如[StockType.TURN_INCREASE, StockType.CONSOLIDATION_INCREASE]
DEBUG_STOCK_TYPE = []
# 指定一个或多个ts_code( ex. ['00000.SZ',''] )
DEBUG_CODE = []
# 是否全部打开debug
DEBUG_ALL = False

# better not greater than 9
pole_normal_window = 9
# todo:magic, should be smaller at the begging or end?
pole_tail_window = 3
pole_head_window = 5
shift_normal = int(pole_normal_window / 2)
shift_tail = int(pole_tail_window / 2)
shift_head = int(pole_head_window / 2)

# todo： magic
# 横盘直线和X轴的角度应该在（-10，10）范围内
CONSOLIDATION_LINE_K_MIN = tan(170)
CONSOLIDATION_LINE_K_MAX = tan(10)
CONSOLIDATION_DAYS_LEAST = 21
CHANGE_DAYS_LEAST = 5
CHANGE_RATIO_LEAST = 0.15
min_data_day = CHANGE_DAYS_LEAST * 4  # 股票历史数据 最小天数
latest_pole_max_distinct = 5  # 最近的点不能太远


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

        start = idx - shift
        if start < 0:
            start = 0
        end = idx + shift + 1
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

        cur_avg = sum(values[idx:end]) / (end - idx)
        # if pre_avg is None:
        pre_avg = sum(values[start:idx + 1]) / (idx + 1 - start)

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


# 固定窗口寻找，判断前一个窗口和当前窗口平均值，决定趋势
# 判断相邻两个趋势是否相同，若一直相同，则为长期增长或下降股票。否则为波动股
def is_long_value_type(detail: DataFrame, meta_data: {}) -> bool:
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
        pre_avg = sum(values[start:idx + 1]) / (idx + 1 - start)

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
        return False

    if cur_increase is True:
        meta_data[DebugKey.STOCK_TYPE] = StockType.LONG_VALUE
    else:
        # todo: change to long value decrease type in future
        return False

    return True


# todo:
def is_market_trade_increase():
    return True


def is_high_quality(intervals: list, changes: list, meta_data: {}) -> bool:
    avg_interval = sum(intervals) / len(intervals)
    avg_change = sum(changes) / len(changes)

    # 相比值变化，不是那么重要。但也要有底线值
    if avg_interval < CHANGE_DAYS_LEAST:
        # meta_data[DebugKey.DISCARD_REASON] = DiscardReason.POLES_INTERVAL_TOO_SMALL
        return False

    # 高低点之间的变动不能太小
    if avg_change < CHANGE_RATIO_LEAST:
        # meta_data[DebugKey.DISCARD_REASON] = DiscardReason.POLES_CHANGE_TOO_SMALL
        return False

    return True


def is_low_quality(intervals: list, changes: list, meta_data: {}) -> bool:
    avg_interval = sum(intervals) / len(intervals)
    avg_change = sum(changes) / len(changes)

    # 相比值变化，不是那么重要。但也要有底线值
    if avg_interval < CHANGE_DAYS_LEAST:
        meta_data[DebugKey.DISCARD_REASON] = DiscardReason.POLES_INTERVAL_TOO_SMALL
        return True

    # 高低点之间的变动不能太小
    if avg_change < CHANGE_RATIO_LEAST:
        meta_data[DebugKey.DISCARD_REASON] = DiscardReason.POLES_CHANGE_TOO_SMALL
        return True

    return False


def is_too_little_count(poles: list, meta_data: {}) -> bool:
    # todo: magic number
    if len(poles) < 3:
        meta_data[DebugKey.DISCARD_REASON] = DiscardReason.TOO_LITTLE_POLES
        return True

    return False


def is_short_selling(poles: list, meta_data: {}) -> bool:
    # latest pole & 做空
    pt = poles[-1][1]
    if pt is PoleType.HIGH:
        # todo: fix to not discard type when ready
        # todo: return stocktype = short_selling
        meta_data[DebugKey.DISCARD_REASON] = DiscardReason.SHORT_SELLING
        return True
    return False


def is_too_far(poles: list, detail: DataFrame, meta_data: {}) -> bool:
    date = detail.iloc[poles[-1][0]]['trade_date']
    date_dist = dict()
    for idx in range(0, len(detail)):
        dt = detail.iloc[idx]['trade_date']
        date_dist[dt] = len(detail) - idx

    # 最近一个点离现在太远了
    if date_dist[date] > latest_pole_max_distinct:
        meta_data[DebugKey.DISCARD_REASON] = DiscardReason.TOO_FAR
        return True

    # is_market_increase = is_market_trade_increase()
    # if is_market_increase and tp == PoleType.HIGH or is_market_increase is False and tp == PoleType.LOW:
    #     return DiscardReason.INVERSE_MARKET

    return False


def is_not_cross_poles(poles: list, meta_data: {}) -> bool:
    # trade date should cross between high/low points
    pre_pt = poles[0][1]
    for idx in range(1, len(poles)):
        pt = poles[idx][1]
        if pt == pre_pt:
            meta_data[DebugKey.DISCARD_REASON] = DiscardReason.SAME_ALIGN_POLES
            return True

        pre_pt = pt
    return False


# 后面的波段是低质量的：涨幅不大或者持续天数不多。用（p2，p1）的历史值来预测下一波段，不一定靠谱
def is_low_quality_stock(poles: list, detail: DataFrame, meta_data: {}) -> bool:
    intervals = []
    changes = []
    for idx in range(1, len(poles) - 1, 2):
        intervals.append(date_diff(detail.iloc[poles[idx][0]]['trade_date'],
                                   detail.iloc[poles[idx + 1][0]]['trade_date']))
        changes.append(abs(detail.iloc[poles[idx][0]][Const.SMOOTH_KEY.name] -
                           detail.iloc[poles[idx + 1][0]][Const.SMOOTH_KEY.name]) /
                       detail.iloc[poles[idx + 1][0]][Const.SMOOTH_KEY.name])

    if is_low_quality(intervals, changes, meta_data) is True:
        return True

    return False


# 判断是否为波段股：波段长度和改变大小都是在相对平稳的值
def is_high_quality_stock(poles: list, detail: DataFrame, meta_data: {}) -> bool:
    # todo: optimize
    intervals = []
    changes = []
    for idx in range(0, len(poles) - 1, 2):
        intervals.append(date_diff(detail.iloc[poles[idx][0]]['trade_date'],
                                   detail.iloc[poles[idx + 1][0]]['trade_date']))
        changes.append(abs(detail.iloc[poles[idx][0]][Const.SMOOTH_KEY.name] -
                           detail.iloc[poles[idx + 1][0]][Const.SMOOTH_KEY.name]) /
                       detail.iloc[poles[idx][0]][Const.SMOOTH_KEY.name])

    if is_high_quality(intervals, changes, meta_data) is not True:
        return False

    intervals.clear()
    changes.clear()
    for idx in range(1, len(poles) - 1, 2):
        intervals.append(date_diff(detail.iloc[poles[idx][0]]['trade_date'],
                                   detail.iloc[poles[idx + 1][0]]['trade_date']))
        changes.append(abs(detail.iloc[poles[idx][0]][Const.SMOOTH_KEY.name] -
                           detail.iloc[poles[idx + 1][0]][Const.SMOOTH_KEY.name]) /
                       detail.iloc[poles[idx][0]][Const.SMOOTH_KEY.name])

    if is_high_quality(intervals, changes, meta_data) is not True:
        return False

    return True


def smooth_values(detail: DataFrame, in_size: int):
    detail[Const.SMOOTH_KEY.name] = (detail['close'] + detail['open']) / 2

    shift = in_size / 2
    for idx in range(0, len(detail)):
        idx_p = detail.iloc[idx][Const.SMOOTH_KEY.name]
        left_n = 1
        left_sum = 0
        while idx - left_n >= 0 and left_n <= shift:
            left_sum += detail.iloc[idx - left_n][Const.SMOOTH_KEY.name]
            left_n += 1

        right_n = 1
        right_sum = 0
        while idx + right_n < len(detail) and right_n <= shift:
            right_sum += detail.iloc[idx + right_n][Const.SMOOTH_KEY.name]
            right_n += 1

        avg = (left_sum + right_sum) / (left_n + right_n - 2)
        detail.loc[idx, Const.SMOOTH_KEY.name] = idx_p + (avg - idx_p) / 2


# 盘整期间
def is_consolidate(poles: list, detail: DataFrame) -> bool:
    if len(poles) < 2:
        return False

    date1 = detail.iloc[poles[-1][0]]['trade_date']
    date2 = detail.iloc[poles[-2][0]]['trade_date']
    day_diff = date_diff(date1, date2)
    if day_diff < CONSOLIDATION_DAYS_LEAST:
        return False

    # 训练数据
    xx = []
    yy = []

    for ii in range(poles[-2][0], poles[-1][0]+1):
        xx.append(ii)
        yy.append(detail.iloc[ii][Const.SMOOTH_KEY.name])

    # 调用拟合函数
    kk, bb = least_squares(xx, yy, len(xx))

    if CONSOLIDATION_LINE_K_MIN < kk < CONSOLIDATION_LINE_K_MAX:
        return True

    return False


# 盘整股，突然上涨
def is_consolidate_increase(poles: list, detail: DataFrame, meta_data: {}) -> bool:
    if is_consolidate(poles, detail) is False:
        return False

    if poles[-1][1] is PoleType.HIGH:
        return False

    return True


# 阶段性回落，又增长。但整体趋势是上涨的
def is_phased_increase(poles: list, detail: DataFrame, meta_data: {}) -> bool:
    if len(poles) < 4:  # todo：需要3个？
        return False

    price_1 = detail.iloc[poles[-1][0]][Const.SMOOTH_KEY.name]
    price_2 = detail.iloc[poles[-2][0]][Const.SMOOTH_KEY.name]
    price_3 = detail.iloc[poles[-3][0]][Const.SMOOTH_KEY.name]
    price_4 = detail.iloc[poles[-4][0]][Const.SMOOTH_KEY.name]

    # todo: magic number. 1.2越大，升的越快
    if price_1 > price_3 * 1.2 and price_2 > price_4 * 1.2:
        return True

    return False


# poles: list of pair
def check_stock_type(poles: list, detail: DataFrame, meta_data: {}) -> bool:
    # must first one
    if filtered_by_score(detail, poles, meta_data):
        meta_data[DebugKey.STOCK_TYPE] = StockType.DISCARD
        return

    if is_long_value_type(detail, meta_data):
        meta_data[DebugKey.STOCK_TYPE] = StockType.LONG_VALUE
        return

    if is_too_little_count(poles, meta_data) \
            or is_too_far(poles, detail, meta_data) \
            or is_short_selling(poles, meta_data) \
            or is_not_cross_poles(poles, meta_data):  # keep?
        meta_data[DebugKey.STOCK_TYPE] = StockType.DISCARD
        return

    # # 做多时不用判断
    # if is_low_quality_stock(poles, detail, meta_data):
    #     meta_data[DebugKey.STOCK_TYPE] = StockType.DISCARD
    #     return

    # 做空时不用判断
    if is_low_quality_stock(poles, detail, meta_data):
        meta_data[DebugKey.STOCK_TYPE] = StockType.DISCARD
        return

    if is_phased_increase(poles, detail, meta_data):
        meta_data[DebugKey.STOCK_TYPE] = StockType.PHASED_INCREASE
        return

    if is_consolidate_increase(poles, detail, meta_data):
        meta_data[DebugKey.STOCK_TYPE] = StockType.CONSOLIDATION_INCREASE
        return

    meta_data[DebugKey.STOCK_TYPE] = StockType.UNKNOWN


# 根据过去一段时间的涨幅来过滤低质量股票，可能跨过波峰波谷
def filtered_by_score1(detail: DataFrame, poles: list, meta_data: {}):
    if DebugKey.STOCK_TYPE in meta_data.keys() and meta_data[DebugKey.STOCK_TYPE] is StockType.DISCARD:
        return

    end = len(detail)
    # TODO: magic number
    start = end - 14
    if len(poles) >= 2:
        start = poles[-2][0]
    if start < 0:
        start = 0

    close_change_pcts = []
    for index in range(start + 1, end):
        close_change_pct = abs(detail.iloc[index]['pct_chg'])  # 今天close相比昨天close变化

        close_change_pcts.append(close_change_pct)

    avg = np.mean(close_change_pcts)
    var = np.var(close_change_pcts)

    meta_data[DebugKey.SCORE_CHANG_DEPTH] = avg
    meta_data[DebugKey.SCORE_CHANG_VAR] = var

    # todo: magic number
    if avg < 1 or var > 3:
        meta_data[DebugKey.DISCARD_REASON] = DiscardReason.SMALL_DEPTH_OR_UNSTABLE
        return True

    return False


# 根据过去一段时间的涨幅来过滤低质量股票，可能跨过波峰波谷
def filtered_by_score(detail: DataFrame, poles: list, meta_data: {}):
    if DebugKey.STOCK_TYPE in meta_data.keys() and meta_data[DebugKey.STOCK_TYPE] is StockType.DISCARD:
        return

    end = len(detail)
    # TODO: magic number
    start = end - 14
    if len(poles) >= 2:
        start = poles[-2][0]
    if start < 0:
        start = 0

    change_pcts = []
    for index in range(start + 1, end):
        change_pct = abs(detail.iloc[index][Const.SMOOTH_KEY.name]-detail.iloc[index-1][Const.SMOOTH_KEY.name])
        change_pcts.append(change_pct)

    avg = np.mean(change_pcts)
    var = np.var(change_pcts)

    meta_data[DebugKey.SCORE_CHANG_DEPTH] = avg
    meta_data[DebugKey.SCORE_CHANG_VAR] = var

    # todo: magic number
    if avg < 0 or var > 1000:
        meta_data[DebugKey.DISCARD_REASON] = DiscardReason.SMALL_DEPTH_OR_UNSTABLE
        return True

    return False


def angle_trunc(a):
    while a < 0.0:
        a += pi * 2
    b = a * 180 / pi
    if b > 270:
        b -= 360
    if -90 < b < 90:
        return b

    print("invalid angle:{0}", b)  # meaningless for stocks


# the angle between x-row and the line: (x_des,y_des)->(x_orig,y_orig)
def get_angle_between_points(x_orig, y_orig, x_des, y_des):
    delta_y = x_des - y_orig
    delta_x = y_des - x_orig
    return angle_trunc(atan2(delta_y, delta_x))


# y = a*x + b
def least_squares(x, y, data_num):
    sumx = 0
    sumx2 = 0
    sumy = 0
    sumxy = 0

    for i in range(0, data_num, 1):
        sumx += x[i]
        sumy += y[i]
        sumx2 += x[i] ** 2
        sumxy += x[i] * y[i]
    k = ((data_num * sumxy) - (sumx * sumy)) / ((data_num * sumx2) - (sumx * sumx))
    b = ((sumx2 * sumy) - (sumx * sumxy)) / ((data_num * sumx2) - (sumx * sumx))

    return k, b


if __name__ == "__main__":

    '''训练数据'''
    x = [0, 2, 4, 6, 8, 10]
    y = [0, 6, 25, 55, 60, 110]

    '''结果输出'''
    x1 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]  # 测试数据
    result = np.zeros((len(x1)), dtype=np.double)  # 结果记录
    '''调用拟合函数'''
    k, b = least_squares(x, y, len(x))
    print("数组长度：", len(x1))
    for i in range(0, len(x1), 1):
        result[i] = k * x1[i] + b

    plt.scatter(x=x, y=y, color="blue")
    plt.plot(x1, result, color="red")
    plt.show()


# ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
def stock_analysis(stock_detail: DataFrame, meta_data: {}):
    if len(stock_detail) < min_data_day:
        meta_data[DebugKey.DISCARD_REASON] = DiscardReason.TOO_SHORT_DAY_DATA
        meta_data[DebugKey.STOCK_TYPE] = StockType.DISCARD
        return

    # for debug
    ts_code = stock_detail.iloc[0]['ts_code']
    if len(DEBUG_CODE) > 0 and str(ts_code) not in DEBUG_CODE:
        meta_data[DebugKey.STOCK_TYPE] = StockType.DISCARD
        meta_data[DebugKey.DISCARD_REASON] = DiscardReason.DEBUG_SKIP
        return

    # todo:magic number
    # savgol_smooth(stock_detail)
    smooth_values(stock_detail, 7)
    smooth_values(stock_detail, 5)
    # smooth_values(stock_detail, 3)

    poles = find_poles(stock_detail, meta_data)

    check_stock_type(poles, stock_detail, meta_data)

    show_graph_pole(stock_detail, poles, meta_data)


def date_diff(d1: str, d2: str):
    time_array1 = time.strptime(d1, "%Y%m%d")
    timestamp_day1 = int(time.mktime(time_array1))
    time_array2 = time.strptime(d2, "%Y%m%d")
    timestamp_day2 = int(time.mktime(time_array2))
    result = (timestamp_day1 - timestamp_day2) // 60 // 60 // 24
    return abs(result)


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


def prepare_detail_4_graph(detail: DataFrame) -> DataFrame:
    detail = detail.rename({'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low', 'trade_date': 'Date',
                            'vol': 'Volume'}, axis='columns')

    detail['Date'] = pd.to_datetime(detail['Date'])
    detail.set_index("Date", inplace=True)

    return detail


def show_kline_graph(detail: DataFrame, meta_data: {}):
    ts_code = detail.iloc[0]['ts_code']
    discard_reason = None
    st = meta_data[DebugKey.STOCK_TYPE]
    if st is StockType.DISCARD:
        discard_reason = meta_data[DebugKey.DISCARD_REASON]
    title = str(ts_code) + str('#') + str(discard_reason) + str('#') + str(st)

    detail = prepare_detail_4_graph(detail)

    # debug for one ts_code
    if str(ts_code) in DEBUG_CODE:
        mpf.plot(detail, type='candle', mav=(3, 6, 9), volume=True, title=title)

    # debug for candidate
    if st in DEBUG_STOCK_TYPE:
        mpf.plot(detail, type='candle', mav=3, volume=False, title=title)

    # debug for all
    if DEBUG_ALL is True:
        mpf.plot(detail, type='candle', mav=3, volume=False, title=title)


def save_pic(candidates: [], folder: str):
    for candidate in candidates:
        score = candidate[0]
        detail = candidate[1]
        ts_code = detail.iloc[0]['ts_code']
        st = detail.iloc[0]['st']
        title = str(score) + '#' + str(ts_code) + '#' + str(st)

        detail = prepare_detail_4_graph(detail)
        mpf.plot(detail, type='candle', mav=3, volume=False, title=title, savefig=folder + title + '.png')


# ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
