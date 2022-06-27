from tigeropen.common.consts import (Language,        # 语言
                                    Market,           # 市场
                                    BarPeriod,        # k线周期
                                    TimelinePeriod,
                                    QuoteRight)       # 复权类型
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.common.util.signature_utils import read_private_key
from tigeropen.quote.quote_client import QuoteClient

def get_client_config(sandbox=False):
    """
    https://www.itiger.com/openapi/info 开发者信息获取
    :return:
    """
    client_config = TigerOpenClientConfig(sandbox_debug=sandbox)
    client_config.private_key = read_private_key('tiger-rsa-private.pem')
    client_config.tiger_id = '20151469'
    client_config.account = 'U8976190'  # 环球账户U码
    client_config.standard_account = None  # 标准账户
    client_config.paper_account = '20210919222113885'  # 模拟账户
    client_config.language = Language.zh_CN
    return client_config


client_config = get_client_config()
quote_client = QuoteClient(client_config)

permissions = quote_client.grab_quote_permission()
#
# list = quote_client.get_market_status(Market.ALL)
# print(list)
#
# print(quote_client.get_symbols(Market.US))
#
#print(quote_client.get_symbol_names(Market.US))

# print(quote_client.get_timeline(['JD'], False))
print(quote_client.get_bars(['JD', '01810', '600730']))
#if __name__ == '__main__':
