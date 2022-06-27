
import time
from pandas import DataFrame
import pandas as pd

pd.set_option('mode.chained_assignment', None)

# parameters
print_switch = False

window_size = 13  # must be odd. 提高此值筛选更大周期的波段股票
ex_threshold = int(window_size / 4)  # 梯度下降提升，异常数量
half_window_size = int(window_size/2)
point_interval_thres = 2.5  #波峰、波谷之间距离和均值之间的误差阈值
point_interval_least = 7
point_change_least = 0.13

high_point_dist = 15
low_point_dist = 4
hot_vol_ratio = 3  # 最近hot_days平均交易量/历史平均交易量
daily_change_thre = 1.5  # 最近每天价格涨幅比例
hot_days = 3  # 保证大于2，最近几天交易量大增，价格上涨
min_hist_days = 20  # 股票历史数据 最小天数


def do_print(content: str):
    if print_switch is False:
        return

    print(content)


# ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
def is_band(stock_detail: DataFrame):

    if filter(stock_detail) is False:
        return False

    date_dist = dict()
    for idx in range(0, len(stock_detail)):
        date = stock_detail.iloc[idx]['trade_date']
        date_dist[date] = idx+1

    low_points = []
    high_points = []

    stock_detail['mid'] = (stock_detail['close'] + stock_detail['open'])/2

    # find candidates
    low_candidate = []
    high_cadidate = []
    le = len(stock_detail)
    for idx in range(half_window_size, le-half_window_size):
        idx_p = stock_detail.iloc[idx]['mid']

        have_low = True
        have_high = True
        i = 1
        while i <= half_window_size:
            if have_low is False and have_high is False:
                break

            if have_low and (stock_detail.iloc[idx-i]['mid'] <= idx_p or
                             stock_detail.iloc[idx+i]['mid'] <= idx_p):
                have_low = False

            if have_high and (stock_detail.iloc[idx-i]['mid'] >= idx_p or
                              stock_detail.iloc[idx+i]['mid'] >= idx_p):
                have_high = False

            i += 1

        if have_low:
            low_candidate.append(idx)
            low_points.append(stock_detail.iloc[idx])
        if have_high:
            high_cadidate.append(idx)
            high_points.append(stock_detail.iloc[idx])
    #
    # do_print('\nbefore filter...')
    # if len(low_points) != 0:
    #     do_print('low:{0},code:{1}'.format(len(low_points), low_points[0]['ts_code']))
    #     dates = []
    #     for date in range(0, len(low_points)):
    #         dates.append(low_points[date]['trade_date'])
    #     do_print('low-dates:{0}'.format(','.join(dates)))
    # if len(high_points) != 0:
    #     do_print('high:{0},code:{1}'.format(len(high_points), high_points[0]['ts_code']))
    #     dates = []
    #     for date in range(0, len(high_points)):
    #         dates.append(high_points[date]['trade_date'])
    #     do_print('high-dates:{0}'.format(','.join(dates)))

    # must be greater than 0
    if len(low_candidate) < 1 or len(high_cadidate) < 1:
        return False

    low_points.clear()
    high_points.clear()

    # double locate precisely
    for idx in range(0, len(low_candidate)):
        if is_point(stock_detail, low_candidate[idx], True):
            row = stock_detail.iloc[low_candidate[idx]]
            row['type'] = 'low'
            low_points.append(row)

    for idx in range(0, len(high_cadidate)):
        if is_point(stock_detail, high_cadidate[idx], False):
            row = stock_detail.iloc[high_cadidate[idx]]
            row['type'] = 'high'
            high_points.append(row)

    if len(low_points) < 1 or len(high_points) < 1:
        return False
    #
    # do_print('\nafter filter...')
    # if len(low_points) != 0:
    #     do_print('low:{0},code:{1}'.format(len(low_points), low_points[0]['ts_code']))
    #     dates = []
    #     for date in range(0, len(low_points)):
    #         dates.append(low_points[date]['trade_date'])
    #     do_print('low-dates:{0}'.format(','.join(dates)))
    # if len(high_points) != 0:
    #     do_print('high:{0},code:{1}'.format(len(high_points), high_points[0]['ts_code']))
    #     dates = []
    #     for date in range(0, len(high_points)):
    #         dates.append(high_points[date]['trade_date'])
    #     do_print('high-dates:{0}'.format(','.join(dates)))

    def get_trade_date(s):
        return s['trade_date']
    points = low_points + high_points
    points.sort(key=get_trade_date, reverse=True)

    # 最近一个点不能离今天太远
    t = points[0]['type']
    date = points[0]['trade_date']
    if t == 'low' and date_dist[date] > low_point_dist or t == 'high' and date_dist[date] > high_point_dist:
        do_print('the latest day is too far!')
        return False

    # trade date should cross between high/low points
    pre_type = points[0]['type']
    for idx in range(1, len(points)):
        tp = points[idx]['type']
        if tp == pre_type:
            do_print('bad cross case!')
            return False
        pre_type = tp

    # interval check logic need this condition st.
    if len(low_points) < 2 and len(high_points) < 2:
        do_print('too little points case!')
        return False

    # interval between two trade dates should be similar
    intervals = []
    changes = []
    mids = []
    for idx in range(0, len(points)-1):
        intervals.append(date_diff(points[idx]['trade_date'], points[idx+1]['trade_date']))
        changes.append(abs(points[idx]['mid'] - points[idx+1]['mid'])/points[idx]['mid'])
        mids.append(abs(points[idx]['mid'] + points[idx+1]['mid'])/2)

    # 整体趋势是上升的
    # TODO:将来也可以考虑下降的
    # for idx in range(1, len(mids)):
    #     if mids[idx] < mids[idx-1]:
    #         do_print('trade not increase!')
    #         return False

    avg_interval = sum(intervals)/len(intervals)
    avg_change = sum(changes) / len(changes)

    # 间距不能太小
    if avg_interval < point_interval_least:
        do_print('interval is too small!')
        return False

    # 高低点之间的变动不能太小
    if avg_change < point_change_least:
        do_print('change is too small!')
        return False

    for idx in range(len(intervals)):
        if abs(intervals[idx] - avg_interval) > point_interval_thres:
            do_print('bad interval case!')
            return False

    print('\nstep and growth...')
    sg = []
    for idx in range(0, len(points)):
        if points[idx]['type'] == 'high':
            continue
        if idx+1 == len(points):
            return

        step = date_diff(points[idx]['trade_date'], points[idx+1]['trade_date'])
        growth = (points[idx+1]['mid']-points[idx]['mid'])*100/points[idx]['mid']
        sg.append('{0}%,{1}day'.format(growth, step))

    print(sg)

    return True


def is_point(stock_detail: DataFrame, idx: int, find_low: bool):
    ex_cnt = 0
    i = 1
    while i <= half_window_size:
        if ex_cnt >= ex_threshold:
            break

        left = idx-i
        right = idx+i
        if find_low:
            if stock_detail.iloc[left]['mid'] <= stock_detail.iloc[left+1]['mid']:
                ex_cnt += 1
            if stock_detail.iloc[right]['mid'] <= stock_detail.iloc[right-1]['mid']:
                ex_cnt += 1
        else:
            if stock_detail.iloc[left]['mid'] >= stock_detail.iloc[left+1]['mid']:
                ex_cnt += 1
            if stock_detail.iloc[right]['mid'] >= stock_detail.iloc[right-1]['mid']:
                ex_cnt += 1

        i += 1

    if ex_cnt < ex_threshold:
        return True


def date_diff(d1: str, d2: str):
    time_array1 = time.strptime(d1, "%Y%m%d")
    timestamp_day1 = int(time.mktime(time_array1))
    time_array2 = time.strptime(d2, "%Y%m%d")
    timestamp_day2 = int(time.mktime(time_array2))
    result = (timestamp_day1 - timestamp_day2) // 60 // 60 // 24
    return result


def filter(detail: DataFrame):
    if len(detail) == 0:
        return False

    price = detail.iloc[0]['open']
    if price > 20:
        return False

    return True


# ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
def is_sharp(stock_detail: DataFrame):
    if stock_detail is None:
        return False

    if filter(stock_detail) is False:
        return False

    detail_len = len(stock_detail)
    if detail_len < min_hist_days:
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
    vols = stock_detail['vol']
    avg_vol_hot = sum(vols[0:hot_days]) / hot_days
    avg_vol_history = sum(vols[hot_days:detail_len]) / detail_len

    if avg_vol_history == 0:
        return False

    if avg_vol_hot / avg_vol_history < hot_vol_ratio:
        return False

    return True
