import base64

import pyodbc

def insertblob(filePath):
    with open(filePath,'rb') as file:
        binarydata = file.read()
        encodestring = base64.b64encode(binarydata)
        blob_value = open('apple2.jpg', 'rb').read()
        sql = 'INSERT INTO image_table(images) VALUES(%s)'
        args = (blob_value,)
        cursor = conn.cursor()
        cursor.execute(sql, args)
        #USP(conn,'[dbo].[insert_image]',['@image'],[encodestring])
def READ(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT*FROM tt')
    for row in cursor:
        print(row)

def USP(conn,PROC_NAME,Param_NAME,PARAM):
    cursor = conn.cursor()
    if(len(Param_NAME)==len(PARAM)):
        storedProc = "Exec " + PROC_NAME
        for i in range(len(Param_NAME)):
            if(i!=len(Param_NAME)-1):
                storedProc=storedProc+" "+Param_NAME[i]+" = "+ str(PARAM[i])+","
            else:
                storedProc = storedProc + " " + Param_NAME[i] + " = " + str(PARAM[i])
        # Execute Stored Procedure With Parameters
        print(storedProc)
        cursor.execute(storedProc)
        conn.commit()
        return True
    else:
        return "Length not equal"


def UPDATE(conn,DB_NAME,Param_NAME,PARAM):
    cursor = conn.cursor()

    cursor.execute(
       "UPDATE " + DB_NAME+ " SET "+Param_NAME[0]+" = "+ str(PARAM[0])+" WHERE "+Param_NAME[0]+" = "+ str(PARAM[0])+";"
    )
    conn.commit()

conn =pyodbc.connect(
    'Driver={SQL Server Native Client 11.0};'
    'Server=DESKTOP-TO7M4FP;'
    'Database=tt;'
    'Trusted_Connection=yes;')

USP(conn,'[dbo].[insert_image_path]',['@imageP', '@imageName'],["'D:/SOFTWARES/PYCHARM/PycharmProjects/SeeFood/archive/chest_xray/train/PNEUMONIA'", "'Pneumonea_Train';"])
USP(conn,'[dbo].[insert_image_path]',['@imageP', '@imageName'],["'D:/SOFTWARES/PYCHARM/PycharmProjects/SeeFood/archive/chest_xray/train/NORMAL'", "'NORMAL_Train';"])
USP(conn,'[dbo].[insert_image_path]',['@imageP', '@imageName'],["'D:/SOFTWARES/PYCHARM/PycharmProjects/SeeFood/archive/chest_xray/test/PNEUMONIA'", "'Pneumonea_Test';"])
USP(conn,'[dbo].[insert_image_path]',['@imageP', '@imageName'],["'D:/SOFTWARES/PYCHARM/PycharmProjects/SeeFood/archive/chest_xray/test/NORMAL'", "'NORMAL_Test';"])
USP(conn,'[dbo].[insert_image_path]',['@imageP', '@imageName'],["'D:/SOFTWARES/PYCHARM/PycharmProjects/SeeFood/archive/chest_xray/val/PNEUMONIA'", "'Pneumonea_Val';"])
USP(conn,'[dbo].[insert_image_path]',['@imageP', '@imageName'],["'D:/SOFTWARES/PYCHARM/PycharmProjects/SeeFood/archive/chest_xray/val/NORMAL'", "'NORMAL_Val';"])

#UPDATE(conn,'Customers',['Email','Name'],['Bleeeko','hamza'])
#UPDATE(conn)
#READ(conn)