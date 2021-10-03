import ocel
from math import isnan


def get_common_attribute_names(log, obj_type=None):
    if obj_type is None:
        obj_type = set(ocel.get_object_types(log))
    else:
        obj_type = set(obj_type)

    attribute_dict = {an: 0.0 for an in ocel.get_attribute_names(log)}
    all_objs = ocel.get_objects(log)
    objs = {o: v for o, v in all_objs.items()
            if all_objs[o]['ocel:type'] in obj_type}

    for o in objs:
        if objs[o]['ocel:type'] in obj_type:
            ovmap = objs[o]['ocel:ovmap']
            for ov in ovmap:
                if ovmap[ov] is not None and (isinstance(ovmap[ov], float)
                   and not isnan(ovmap[ov])):
                    attribute_dict[ov] = attribute_dict[ov] + (1 / len(objs))

    return {an: av for an, av in attribute_dict.items() if av != 0.0}
