from onpolicy.utils.format_logger import AppLogger

logger = AppLogger().get_logger()


class AbstractEntry:
    def __init__(self, config):
        self.config = config
        self.id = self._get_unique_id()

    def get_id(self):
        return self.id

    def remove_self_from_list(self, list_parameter):

        set_none_id = -1
        for i, item in enumerate(list_parameter):
            if self is item:
                set_none_id = i
                break
        assert set_none_id != -1, "self not found"
        logger.info("设置了列表中的无人机为None")
        list_parameter[set_none_id] = None

    def _get_unique_id(self):
        return str(self).split('at')[1].strip()[:-1]


if __name__ == '__main__':
    a = AbstractEntry(None)
    test_list = [a, AbstractEntry(None), AbstractEntry(None), AbstractEntry(None), AbstractEntry(None)]
    print("移除前：" + str(test_list))
    a.remove_self_from_list(test_list)
    print("移除后：" + str(test_list))
