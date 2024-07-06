def input_getting(input_type : type,
                  input_message = "",
                  low_bound = None,
                  up_bound = None,
                  answers : list = None):
    input_data = input(input_message)
    try:
        performed_data = input_type(input_data)
        if low_bound != None and up_bound != None:
            assert performed_data >= low_bound and performed_data <= up_bound
        if answers != None:
            assert performed_data in answers
    except:
        return input_getting(input_type = input_type,
                      input_message = input_message,
                      low_bound = low_bound,
                      up_bound = up_bound)
    return performed_data


