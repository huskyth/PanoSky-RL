class ObjectTool:
    @staticmethod
    def set_value_by_name(instance, key, value):
        setattr(instance, key, value)

    @staticmethod
    def get_value_by_name(instance, name):
        return getattr(instance, name)
