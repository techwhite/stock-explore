import commonV2 as lib

import tushare as ts
import datetime
from pandas import DataFrame
import pandas as pd

pd.set_option('mode.chained_assignment', None)

history_days = 60  # 获取股票历史数据天数
batch_call_count = int(5000 * 0.2 / history_days)


now = datetime.datetime.now()
end_date = (now + datetime.timedelta(days=-1)).strftime("%Y%m%d")
start_date = (now + datetime.timedelta(days=-history_days)).strftime("%Y%m%d")
pro = ts.pro_api('af70725c78bbeb6c166a104f735ecc0f62db1348a4b8533812621554')


# batch call
def do_query(ts_codes: str):
    return pro.query('daily', ts_code=ts_codes, start_date=start_date, end_date=end_date)


# sub find
def sub_find(sub_stocks: DataFrame):
    code_list = sub_stocks['ts_code'].tolist()
    ts_codes = ','.join(code_list)

    detail_map = do_query(ts_codes=ts_codes).groupby('ts_code')
    keys = set()
    for key, group in detail_map:
        keys.add(key)

    # 输出
    # detail_map.apply(print)

    sub_candidates = []
    for ts_code in code_list:
        if str(ts_code).startswith('3'):
            # 创业板暂不支持
            continue

        if ts_code not in keys:
            continue

        detail = detail_map.get_group(ts_code)
        # sort by trade date and reindex
        detail = detail.sort_values(by='trade_date')
        new_detail = detail.reset_index()

        meta_data = {}
        st = lib.stock_analysis(new_detail, meta_data)
        if st != lib.StockType.BAND_INCREASE:
            continue

        # todo: fix
        candidate = sub_stocks.loc[sub_stocks['ts_code'] == ts_code]
        candidate['type'] = 'sharp'
        print('\nsharp stock:')
        print(candidate)
        sub_candidates.append(candidate)

    return sub_candidates


def find():
    stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    candiates = []

    length = len(stocks)
    idx = 0
    while idx < length:
        end_idx = idx + batch_call_count
        if idx + batch_call_count > length:
            end_idx = length

        sub_stocks = stocks[idx:end_idx]
        sub_candiates = sub_find(sub_stocks)
        if len(sub_candiates) != 0:
            candiates += sub_candiates

        idx = end_idx

    return candiates


if __name__ == '__main__':
    stocks = find()
    # stocks = do_query('600170.SH')
    print(stocks)

