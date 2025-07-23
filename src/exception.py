import sys

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error_messsage = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,line_no,str(error)
    )
    return Exception

    class CustomException(Exception):
        def __init__(self, error_message, error_detail:sys):
            super.__init__(error_message)
            self.error_message = error_messsage_detail(error_messsage,error_detail=error_detail)
            
        def __str__(self):
            return self.error_message