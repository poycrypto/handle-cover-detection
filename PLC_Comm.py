from main import *


def read_bool(plc,db_num,start_byte,boolean_index):
    try:
        data = plc.db_read(db_num, start_byte, 1)
        bool_val = snap7.util.get_bool(data, 0, boolean_index)
        return bool_val
    except:
        pass

def read_int(plc,db_num,start_byte):
    try:
        data = plc.db_read(db_num,start_byte,2)
        int_val = snap7.util.get_int(data,0)
        return int_val
    except:
        pass

def write_bool(plc,db_num,start_byte,boolean_index,bool_value): #Bool yazma 
    try:
        data = bytearray(1)
        snap7.util.set_bool(data,0,boolean_index,bool_value)
        plc.db_write(db_num,start_byte,data)
    except:
        pass

def write_byte(plc,db_num,start_byte,byte_value): #Byte yazma 
    try:
        data = bytearray(1)
        snap7.util.set_byte(data,0,byte_value)
        plc.db_write(db_num,start_byte,data)
    except:
        pass

def write_int(plc,db_num,start_byte,int_value): #Integer yazma 
    try:
        data = bytearray(2)
        snap7.util.set_int(data,0,int_value)
        plc.db_write(db_num,start_byte,data)
    except:
        pass

def write_real(plc,db_num,start_byte,real_value): #Real yazma 
    try:
        data = bytearray(4)
        snap7.util.set_real(data,0,real_value)
        plc.db_write(db_num,start_byte,data)
    except:
        pass