from collections import namedtuple
from common.time_utils import get_current_time
from dao.dynamo_dao import DynamoDAO
import msgpack
import itertools

TPFYResult = namedtuple("TPFYResult", ["dw_p_id", "result", "updated_at"])


class TFPYDDynamoDB(DynamoDAO):
    def __init__(self, table):
        super().__init__(table)

    @staticmethod
    def pack_item(
        p_id,
        home_result,
        decode=False,
    ):
        if len(home_result) == 0:
            return None

        row = {
            "p_id": {"S": p_id},
            "result": {"L": [{"S": str(c)} for c, _ in home_result]},
            "updated_at": {"N": str(get_current_time())},
            "ttl": {"N": str(get_current_time() + 86400 * 365 * 2)},
        }

        return row

    # @staticmethod
    # def unpack_item(response):
    #     pid = response['p_id']['S'] if 'p_id' in response else None
    #     result = []
    #     if 'result' in response:
    #         result = [str(x['S']) for x in response['result']['L']]
    #     updated_at = (response['updated_at']['N']
    #                   if 'updated_at' in response else 0)
    #     return TPFYResult(pid, result, updated_at)

    def batch_put_generator(self, result_iter, pack_fn):
        return self._batch_write_iter(result_iter, pack_fn=pack_fn)

    # def scan(self, limit=None):
    #     count = 0
    #     items, last_key = self._scan()
    #     while limit is None or count < limit:
    #         for item in items:
    #             count += 1
    #             if count % 10000 == 0:
    #                 print('scan processed: ', count)
    #             yield self.unpack_item(item)
    #
    #         if last_key is not None:
    #             items, last_key = self._scan(start_key=last_key)
    #         else:
    #             print('scan finishes: ', count)
    #             break
    #     return
