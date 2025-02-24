from collections import OrderedDict

## Instead of sql.py, we made our own parse function to convert predicted results into predicted slot values
def custom_pred_parse(pred):
    # parse predicted result results and fix general errors

    # fix for no states
    if pred == "":
        return {}

    pred_slot_values = {}

    slot_value = pred.split(",")

    value_assigner = "="
    for i in slot_value:
      if value_assigner not in i:
        continue
      else:
        pred_slot_values[i.split(value_assigner)[0].strip()] = i.split(value_assigner)[1].strip()

    return pred_slot_values



def custom_pred_cot_parse(pred):

    # the output format is "(slot_name = value)"
    start_pos = pred.rfind("(") + 1
    end_pos = pred.rfind(")")
    pred = pred[start_pos:end_pos]

    # fix for no states
    if pred == "":
        return {}

    pred_slot_values = {}

    slot_value = pred.split(",")

    value_assigner = "="
    for i in slot_value:
      if value_assigner not in i:
        continue
      else:
        pred_slot_values[i.split(value_assigner)[0].strip()] = i.split(value_assigner)[1].strip()

    return 



def sv_dict_to_string(svs, sep=' ', sort=True):
    result_list = [f"{s.replace('-', sep)}{sep}{v}" for s, v in svs.items()]
    if sort:
        result_list = sorted(result_list)
    return ', '.join(result_list)
