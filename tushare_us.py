import common as lib

import pandas as pd
import tushare as ts
import datetime
import time
from pandas import DataFrame

# parameters
history_days = 40  # 获取股票历史数据天数
batch_call_count = 100  # us api 限制最多100个

now = datetime.datetime.now()
end_date = (now + datetime.timedelta(days=-1)).strftime("%Y%m%d")
start_date = (now + datetime.timedelta(days=-history_days)).strftime("%Y%m%d")
pro = ts.pro_api('af70725c78bbeb6c166a104f735ecc0f62db1348a4b8533812621554')


# batch call
def do_query(ts_codes: str):
    time.sleep(30)

    # classify = ADR/GDR/EQ
    return pro.us_daily(ts_code=ts_codes, classify='ADR', start_date=start_date, end_date=end_date)


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
        if ts_code not in keys:
            continue

        detail = (DataFrame)(detail_map.get_group(ts_code))
        detail = detail.rename({'pct_change': 'pct_chg'}, axis='columns')
        if lib.is_sharp(detail):
            candidate = sub_stocks.loc[sub_stocks['ts_code'] == ts_code]
            candidate['type'] = 'sharp'
            print('\nsharp stock:')
            print(candidate)
            sub_candidates.append(candidate)

        if lib.is_band(detail):
            candidate = sub_stocks.loc[sub_stocks['ts_code'] == ts_code]
            candidate['type'] = 'band'
            print('\nband stock:')
            print(candidate)
            sub_candidates.append(candidate)

    return sub_candidates


def find():
    # ts_code,name,enname,classify,list_date,delist_date
    columns = None
    data = []
    offset = 0
    while True:
        temp = DataFrame(pro.us_basic(offset=offset))
        if columns is None:
            columns = temp.columns.tolist()

        for idx in range(0, len(temp)):
            row = temp.iloc[idx]
            list = row.tolist()
            delist_date = row['delist_date']
            if delist_date is None:
                data.append(list)

        ln = len(temp)
        if ln < 6000:
            break
        offset += ln
        time.sleep(30)

    stocks = pd.DataFrame(data, columns=columns)

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
